"""Redis-based reliable queue for LLM request processing."""

import json
import logging
import asyncio
from typing import Optional, Tuple, Any
from datetime import datetime
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Constants
JOB_PROCESSING_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_MAX_RETRIES = 3


class RedisQueue:
    """Reliable Redis queue with job acknowledgement and dead-letter handling."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = "redis_password",
        db: int = 0,
        queue_name: str = "llm_requests",
        processing_set: str = "llm_processing_ids",
        jobs_hash: str = "llm_jobs",
        dlq_name: str = "llm_dlq",
        job_ttl: int = JOB_PROCESSING_TIMEOUT_SECONDS,
    ):
        """Initialize Redis queue.

        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Redis database number
            queue_name: Main queue name (sorted set)
            processing_set: Processing set for in-flight job IDs
            jobs_hash: Hash storing job metadata by job_id
            dlq_name: Dead letter queue name
            job_ttl: Job TTL in seconds (for auto-retry of stalled jobs)
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.queue_name = queue_name
        self.processing_set = processing_set
        self.jobs_hash = jobs_hash
        self.dlq_name = dlq_name
        self.job_ttl = job_ttl
        self.redis: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self.redis = await redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            # Test connection
            await self.redis.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")

    async def enqueue(
        self,
        job_id: str,
        model: str,
        request_data: dict,
        job_type: str = "completion",
        priority: int = 0
    ) -> None:
        """Enqueue a job with optional priority (higher = earlier).

        Args:
            job_id: Unique job identifier
            model: Model name
            request_data: Request data dict
            job_type: Job type (completion, embedding, etc.)
            priority: Priority level (0 is default)
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        job_payload = json.dumps({
            "job_id": job_id,
            "model": model,
            "request_data": request_data,
            "job_type": job_type,
            "timestamp": datetime.utcnow().isoformat(),
            "retries": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
        })

        # Use sorted set for priority queue
        score = -priority  # Negative so higher priority is first
        await self.redis.zadd(self.queue_name, {job_payload: score})
        logger.info(f"Enqueued job {job_id} for model {model}")

    async def dequeue_with_lock(self) -> Optional[Tuple[str, dict]]:
        """Dequeue job atomically with lock using BZPOPMIN.

        Returns:
            Tuple of (job_id, job_dict) or None if queue empty
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        try:
            # BZPOPMIN: blocks until item available (1s timeout)
            result = await self.redis.bzpopmin(self.queue_name, timeout=1)
            if not result:
                return None

            # result is (key, member, score)
            job_json = result[1]  # member (the job payload)
            job_dict = json.loads(job_json)
            job_id = job_dict["job_id"]

            # Store full job in hash with processing timestamp
            job_with_timestamp = {
                **job_dict,
                "processing_started": datetime.utcnow().isoformat(),
            }
            await self.redis.hset(self.jobs_hash, job_id, json.dumps(job_with_timestamp))

            # Add to processing set
            await self.redis.sadd(self.processing_set, job_id)

            # Note: Per-job TTL is enforced by cleanup_stalled_jobs() using processing_started timestamps
            # Redis hash fields don't support individual TTLs, so we don't use EXPIRE here

            logger.info(f"Dequeued job {job_id} from queue")
            # Return job_id (used for acknowledgement) and job_dict
            return (job_id, job_dict)

        except Exception as e:
            logger.error(f"Error dequeuing job: {e}")
            return None

    async def acknowledge_job(self, job_id: str) -> None:
        """Remove job from processing (success).

        Args:
            job_id: Job ID from dequeue_with_lock
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        try:
            # Remove from processing set
            await self.redis.srem(self.processing_set, job_id)
            # Remove from jobs hash
            await self.redis.hdel(self.jobs_hash, job_id)
            logger.info(f"Acknowledged job {job_id}")
        except Exception as e:
            logger.error(f"Error acknowledging job: {e}")

    async def nack_job(self, job_id: str, reason: str = "unknown") -> None:
        """Return job to queue (failure, with retry logic).

        Args:
            job_id: Job ID from dequeue_with_lock
            reason: Reason for nack
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        try:
            # Get job from hash
            job_json = await self.redis.hget(self.jobs_hash, job_id)
            if not job_json:
                logger.warning(f"Job {job_id} not found in hash during NACK, removing from processing set")
                await self.redis.srem(self.processing_set, job_id)
                return

            job_dict = json.loads(job_json)
            job_dict["retries"] = job_dict.get("retries", 0) + 1
            max_retries = job_dict.get("max_retries", DEFAULT_MAX_RETRIES)

            # Remove from processing
            await self.redis.srem(self.processing_set, job_id)
            await self.redis.hdel(self.jobs_hash, job_id)

            if job_dict["retries"] >= max_retries:
                # Move to DLQ
                dlq_entry = json.dumps({
                    **job_dict,
                    "dlq_reason": reason,
                    "dlq_timestamp": datetime.utcnow().isoformat(),
                })
                await self.redis.lpush(self.dlq_name, dlq_entry)
                logger.warning(
                    f"Job {job_id} moved to DLQ after "
                    f"{job_dict['retries']} retries: {reason}"
                )
            else:
                # Re-enqueue with updated retry count
                updated_job = json.dumps(job_dict)
                score = job_dict.get("priority", 0)
                await self.redis.zadd(self.queue_name, {updated_job: -score})
                logger.info(
                    f"Re-queued job {job_id} "
                    f"(retry {job_dict['retries']}/{max_retries}): {reason}"
                )

        except redis.ConnectionError as e:
            logger.critical(f"Redis connection error during NACK for job {job_id}: {e}")
            raise RuntimeError("Queue service unavailable during job NACK")
        except Exception as e:
            logger.error(f"Error nacking job {job_id}: {e}", exc_info=True)
            raise

    async def get_queue_stats(self) -> dict:
        """Get current queue statistics."""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        try:
            queue_size = await self.redis.zcard(self.queue_name)
            processing_size = await self.redis.scard(self.processing_set)
            dlq_size = await self.redis.llen(self.dlq_name)

            return {
                "pending": queue_size,
                "processing": processing_size,
                "dead_letter_queue": dlq_size,
                "total": queue_size + processing_size + dlq_size,
            }
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {
                "pending": 0,
                "processing": 0,
                "dead_letter_queue": 0,
                "total": 0,
            }

    async def cleanup_stalled_jobs(self) -> int:
        """Move stalled jobs from processing set back to queue.

        Returns:
            Number of jobs recovered
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        try:
            recovered = 0

            # Get all processing job IDs
            processing_job_ids = await self.redis.smembers(self.processing_set)

            for job_id in processing_job_ids:
                job_json = await self.redis.hget(self.jobs_hash, job_id)
                if not job_json:
                    logger.warning(f"Job {job_id} in processing set but not in hash, cleaning up orphan")
                    await self.redis.srem(self.processing_set, job_id)
                    recovered += 1
                    continue

                job_dict = json.loads(job_json)
                processing_started = datetime.fromisoformat(
                    job_dict.get("processing_started", datetime.utcnow().isoformat())
                )

                # If job has been processing > TTL, recover it
                if (datetime.utcnow() - processing_started).total_seconds() > self.job_ttl:
                    await self.nack_job(job_id, "stalled_job_timeout")
                    recovered += 1

            if recovered > 0:
                logger.warning(f"Recovered {recovered} stalled jobs from processing set")

            return recovered

        except Exception as e:
            logger.error(f"Error cleaning up stalled jobs: {e}")
            return 0

    async def get_dlq_jobs(self, limit: int = 10) -> list:
        """Peek at dead-letter queue jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of jobs in DLQ
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        try:
            jobs = await self.redis.lrange(self.dlq_name, 0, limit - 1)
            return [json.loads(job) for job in jobs]
        except Exception as e:
            logger.error(f"Error getting DLQ jobs: {e}")
            return []

    async def clear_dlq(self) -> int:
        """Clear all jobs from dead-letter queue.

        Returns:
            Number of jobs deleted
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        try:
            count = await self.redis.llen(self.dlq_name)
            await self.redis.delete(self.dlq_name)
            logger.info(f"Cleared {count} jobs from DLQ")
            return count
        except Exception as e:
            logger.error(f"Error clearing DLQ: {e}")
            return 0
