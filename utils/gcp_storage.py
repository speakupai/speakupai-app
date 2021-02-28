import os
import sys
import datetime
import requests
import google.auth

# [START storage_upload_file]
from google.cloud import storage

from google.auth.transport import requests
from google.auth import compute_engine

def upload_blob_and_generate_url(bucket_name, source_file_name, destination_blob_name):

    # auth_request = requests.Request()
    # credentials, project = google.auth.default()
    # signing_credentials = compute_engine.IDTokenCredentials(auth_request, "", service_account_email=credentials.service_account_email)


    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    #alternative storage_client = storage.Client.from_service_account_json('/path/to/service_account_key.json')
    #when an instance on GCP is assocaited with a service account, it will automatically detect Application default credentials: GOOGLE_APPLICATION_CREDENTIALS
    # storage_client = storage.Client(project, credentials)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    try:
        blob.upload_from_filename(source_file_name)

        url = blob.generate_signed_url(
            version="v4",
            # This URL is valid for 15 minutes
            expiration=datetime.timedelta(minutes=15),
            # Allow GET requests using this URL.
            method="GET",
            # credentials=signing_credentials,
        )

        if url:
            os.remove(source_file_name)

    except Exception as e:
        url = str(e)
    
    print(url)
    return url



if __name__ == "__main__":
    upload_blob_and_generate_url(
        bucket_name=sys.argv[1],
        source_file_name=sys.argv[2],
        destination_blob_name=sys.argv[3],
    )