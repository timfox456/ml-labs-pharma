
#############################################################
#
# This Lambda function is written to a file by the notebook 
# It does not run in the notebook!
#
#############################################################

import json
import boto3
import uuid
import os

s3_client = boto3.client('s3')
transcribe_client = boto3.client('transcribe', region_name='us-west-2')

def lambda_handler(event, context):
    # Extract the bucket name and key from the incoming event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # One of a few different checks to ensure we don't end up in a recursive loop.
    if key != "dialog.mp3": 
        print("This demo only works with dialog.mp3.")
        return

    try:
        
        job_name = 'transcription-job-' + str(uuid.uuid4()) # Needs to be a unique name

        response = transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': f's3://{bucket}/{key}'},
            MediaFormat='mp3',
            LanguageCode='en-US',
            OutputBucketName= os.environ['S3BUCKETNAMETEXT'],  # specify the output bucket
            OutputKey=f'{job_name}-transcript.json',
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 2
            }
        )
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error occurred: {e}")
        }

    return {
        'statusCode': 200,
        'body': json.dumps(f"Submitted transcription job for {key} from bucket {bucket}.")
    }

