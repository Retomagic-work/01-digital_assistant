# Use an official Python runtime as a parent image
FROM nvcr.io/nvidia/pytorch:23.11-py3

# Set the working directory in the container
WORKDIR /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir pika python-dotenv

#RUN pip -q install git+https://github.com/huggingface/transformers # need to install from github
RUN pip -q install transformers[sentencepiece]

#RUN pip install -q datasets loralib sentencepiece
#RUN pip -q install bitsandbytes accelerate xformers einops
#RUN pip -q install langchain
RUN pip -q install sacremoses
#RUN pip -q install nltk
#RUN pip -q install tensorflow
RUN pip -q install ctransformers[cuda]

# Download pretrained models
ADD download_models.py /usr/src/app
RUN python download_models.py

ADD intents.json /usr/src/app
#ADD training.py /usr/src/app
#RUN python training.py
#ADD basic_chatbot.py /usr/src/app


# Add only the necessary files
ADD assistant.py /usr/src/app
ADD app.py /usr/src/app
ADD .env /usr/src/app



# Run app.py when the container launches
CMD ["python", "app.py"]
