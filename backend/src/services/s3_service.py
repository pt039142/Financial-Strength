"""
AWS S3 integration for file storage
"""

import boto3
from src.core.config import settings


class S3Service:
    """Service for AWS S3 file operations"""

    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        self.bucket = settings.AWS_S3_BUCKET

    async def upload_file(
        self,
        file_content: bytes,
        file_name: str,
        folder: str = "uploads",
    ) -> str:
        """
        Upload file to S3.
        
        Returns: S3 object key
        """
        key = f"{folder}/{file_name}"
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=file_content,
            )
            return key
        except Exception as e:
            raise Exception(f"Failed to upload file: {e}")

    async def download_file(self, key: str) -> bytes:
        """Download file from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except Exception as e:
            raise Exception(f"Failed to download file: {e}")

    async def delete_file(self, key: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            raise Exception(f"Failed to delete file: {e}")

    def get_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """Generate presigned URL for accessing S3 file"""
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expiration,
            )
            return url
        except Exception as e:
            raise Exception(f"Failed to generate presigned URL: {e}")
