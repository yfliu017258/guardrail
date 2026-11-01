class TargetError(Exception):
    """Retryable transport / server failure (not a decision). The runner retries these with backoff;
    any other exception is recorded on the row as a non-retryable error."""
