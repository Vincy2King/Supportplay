FROM python:3.9
WORKDIR /app
#RUN apt update && apt install -y libgl1-mesa-glx && gcc cmake
RUN apt update && apt install -y libgl1-mesa-glx gcc cmake
RUN apt-get install festival espeak-ng -y
RUN pip install dlib
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
#RUN git clone https://github.com/davisking/dlib.git
#RUN cd dlib
#RUN mkdir build155.94.204.194
#RUN cd build
#RUN apt install -y cmake
#RUN cmake ..
#RUN make
#RUN cd ../../
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
CMD ["python", "app.py"]
