import pika
from pika import PlainCredentials
from assistant import DigitalAssistant
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

assistant = DigitalAssistant("Helsinki-NLP/opus-mt-ru-en", "Helsinki-NLP/opus-mt-en-ru")

def get_LLM_response(request_text):
    try:
        response = assistant.process_text(request_text)
        return f"{response}", 0.9
    except Exception as e:
        print(f"Error in processing text: {e}")
        return "Error in processing request", 0

# Define your credentials
credentials = PlainCredentials(os.getenv('RABBITMQ_USER'), os.getenv('RABBITMQ_PASSWORD'))  

while True:
    try:
        # Establish a connection with credentials  
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(os.getenv('RABBITMQ_HOST'), os.getenv('RABBITMQ_PORT'), '/', credentials=credentials)
        )

        # Create a channel
        channel = connection.channel()

        # Declare a queue
        channel.queue_declare(queue='requests')

        # Create a callback function to process messages
        def callback(ch, method, properties, body):
            try:
                print(f" [x] Received {body}")
                message_dict = json.loads(body.decode('utf-8'))
                print(message_dict)
                request_text = message_dict["data"]["text"]  # Access "text" field
                request_id = message_dict["id"]
                
                neural_network_response, confidence = get_LLM_response(request_text)
                
                response_dict = {
                    "id": request_id,
                    "data": {
                        'text': neural_network_response,
                        'confidence': confidence
                    }
                }
                print(response_dict)
                
                response_dumped = json.dumps(response_dict)
                
                channel.basic_publish(exchange='', routing_key='responses', body=response_dumped)  
                
                print(f" [x] Sent:{response_dumped}")
            except Exception as e:
                print(f"Error in processing message: {e}")

        # Set up consumer
        channel.basic_consume(queue='requests', on_message_callback=callback, auto_ack=True)

        print(' [*] Waiting for messages. To exit press CTRL+C')

        # Start consuming
        channel.start_consuming()

    except KeyboardInterrupt:
        channel.stop_consuming()
        break

    except Exception as e:
        print(f"Error in message consumption: {e}")
        print("Attempting to reconnect...")
