FROM python:3.10-slim

RUN pip3 install ovos-utils~=0.0.38 ovos-plugin-manager==0.0.25 ovos-tts-server~=0.0.3a10

COPY . /tmp/ovos-tts-plugin-piper
RUN pip3 install /tmp/ovos-tts-plugin-piper

ENTRYPOINT ovos-tts-server --engine ovos-tts-plugin-piper
