import os
import json
import time
import boto3
import logging

logging.basicConfig(level=logging.INFO)

BUCKET_NAME=""
BUCKET_REGION=""
AWS_SECRET_KEY_ID=""
AWS_SECREY_ACCESS_KEY=""
QUEUE_NAME=""

s3 = boto3.client('s3',
                    aws_access_key_id=AWS_SECRET_KEY_ID,
                    aws_secret_access_key=AWS_SECREY_ACCESS_KEY,
                    region_name=BUCKET_REGION,
)
sqs = boto3.resource('sqs',
                     aws_access_key_id=AWS_SECRET_KEY_ID,
                     aws_secret_access_key=AWS_SECREY_ACCESS_KEY,
                     region_name=BUCKET_REGION,
)
queue = sqs.get_queue_by_name(QueueName=QUEUE_NAME)

def download_model():
    MODEL_NAME="256x256_diffusion_uncond.pt"
    if not os.path.isfile(f"./{MODEL_NAME}"):
        s3.download_file(BUCKET_NAME, f"engine/{MODEL_NAME}", f"./{MODEL_NAME}")

def process_message():
    for message in queue.receive_messages(MaxNumberOfMessages=1):
        try:
            print(message.body)
            body = json.loads(message.body)
            prompt = body["prompt"]
            logging.info(f"starting job with prompt: {prompt}")

            from generate_diffuse import args, do_run
            do_run(args, [prompt])

            store_results(body, "./output.png")

            message.delete()
        except Exception as e:
            logging.error(e)   

def store_results(data, image_path):
    key = data["jobId"]
    s3.put_object(
        Body=json.dumps(data),
        Bucket=BUCKET_NAME,
        Key=f"data/{key}.json"
    )
    s3.upload_file(image_path, Bucket=BUCKET_NAME, Key=f"images/{key}.png")

if __name__=="__main__":
    download_model()
    while True:
        process_message()
        time.sleep(20)
