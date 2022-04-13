import os
import json
import time
import boto3
import logging

logging.basicConfig(level=logging.INFO)

BUCKET_NAME=""
BUCKET_REGION=""
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
QUEUE_NAME=""

s3 = boto3.client('s3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=BUCKET_REGION,
)
sqs = boto3.resource('sqs',
                     aws_access_key_id=AWS_ACCESS_KEY_ID,
                     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                     region_name=BUCKET_REGION,
)
queue = sqs.get_queue_by_name(QueueName=QUEUE_NAME)

def download_model():
    # MODEL_NAME="256x256_diffusion_uncond.pt"
    MODEL_NAME="512x512_diffusion_uncond_finetune_008100.pt"
    if not os.path.isfile(f"./{MODEL_NAME}"):
        s3.download_file(BUCKET_NAME, f"engine/{MODEL_NAME}", f"./{MODEL_NAME}")

def process_message():
    for message in queue.receive_messages(MaxNumberOfMessages=1):
        try:
            print(message.body)
            data = json.loads(message.body)
            prompt = data["prompt"]
            logging.info(f"starting job with prompt: {prompt}")
            store_status(data, "processing")

            from generate_diffuse import args, do_run
            do_run(args, [prompt])

            store_results(data, "./output.png")

            message.delete()
        except Exception as e:
            store_status(data, "exception")
            logging.error(e)

def store_status(data, status):
    key = data["jobId"]
    data["status"] = status
    s3.put_object(
        Body=json.dumps(data),
        Bucket=BUCKET_NAME,
        Key=f"data/{key}.json"
    )

def store_results(data, image_path):
    key = data["jobId"]
    data["status"] = "done"
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
