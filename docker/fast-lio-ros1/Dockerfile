# Imagem do FAST-LIO (ROS 1 Noetic) usada pela implantação do backend FAST-LIO.
#
# Esta receita descreve a imagem `contextual-3d-mapping-fastlio-ros1`, cujo build local
# (ID sha256:670973462caa9496acaf41dd2a94f07277f14439b4d026d3bd9965c81b1dd985, 2026-09-09,
# feito pela forma anterior desta receita, sem base nem pacotes APT fixados) rodou a
# validação real do backend. A equivalência dessa imagem com esta receita foi verificada
# (ver src/contextmap/state_estimation/docs/backends.md).
#
# A imagem contém só o FAST-LIO compilado; o wrapper
# (src/contextmap/state_estimation/backends/fast_lio_wrapper.py) é montado em tempo de
# execução, então alterá-lo nunca exige reconstruir a imagem.
#
#   docker build -t contextmap-fast-lio-ros1 docker/fast-lio-ros1
#
# O que esta receita FIXA:
#   - a imagem base, pelo digest do manifesto (a tag `noetic-desktop-full` é móvel);
#   - todo pacote APT que o passo de instalação nomeia, por versão exata (o passo só instala
#     git, git-man e liberror-perl; os demais já vêm da base e a versão é só confirmada);
#   - o Livox SDK, o driver livox_ros_driver e o FAST-LIO, por commit (os submódulos do
#     FAST-LIO vêm dos gitlinks desse commit).
#
# O que NÃO é fixado, e portanto o build não é garantidamente idêntico:
#   - a disponibilidade dos artefatos fixados: se o arquivo do Ubuntu deixar de servir uma
#     das versões APT, ou o GitHub um dos três repositórios, o build FALHA (não troca de
#     versão em silêncio), mas deixa de ser reconstruível;
#   - o binário: o CMakeLists do FAST-LIO grava em tempo de compilação o número de threads do
#     OpenMP a partir do número de processadores da máquina de build (`ProcessorCount`), e o
#     build não é bit a bit reproduzível (não há SOURCE_DATE_EPOCH nem flags determinísticas);
#   - por isso o ID da imagem não é o critério de identidade: o digest da base, a lista de
#     pacotes e o hash do binário `fastlio_mapping` são (ver backends.md).
#
# O FAST-LIO é GPL-2.0 (hku-mars/FAST_LIO); é compilado e executado aqui, não redistribuído.
FROM osrf/ros:noetic-desktop-full@sha256:7dbfb9576d8e6d226c31e06129a82aaab8702695f38eca2116918cb9b9308797

SHELL ["/bin/bash", "-c"]
ARG FAST_LIO_REF=7cc4175de6f8ba2edf34bab02a42195b141027e9
ARG LIVOX_SDK_REF=9306596a2bf15c1343bc023b497465ed0a32909d
ARG LIVOX_ROS_DRIVER_REF=3d240d5666129e1a3052e78ee8487a04b08fdda3

# Versões lidas com dpkg-query na imagem validada. Um pacote sem `=versão` falharia o teste
# tests/state_estimation/test_state_estimation_fast_lio_image_recipe.py.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.8ubuntu1.1 \
        cmake=3.16.3-1ubuntu1.20.04.1 \
        git=1:2.25.1-1ubuntu3.14 \
        git-man=1:2.25.1-1ubuntu3.14 \
        libapr1-dev=1.6.5-1ubuntu1.1 \
        liberror-perl=0.17029-1 \
    && rm -rf /var/lib/apt/lists/*

# As mensagens e o CMake do FAST-LIO dependem do driver Livox mesmo para outros tipos de LiDAR.
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git /tmp/Livox-SDK \
    && git -C /tmp/Livox-SDK checkout --detach "${LIVOX_SDK_REF}" \
    && cmake -S /tmp/Livox-SDK -B /tmp/Livox-SDK/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/Livox-SDK/build --parallel \
    && cmake --install /tmp/Livox-SDK/build \
    && rm -rf /tmp/Livox-SDK

RUN mkdir -p /opt/fast-lio/src \
    && git clone https://github.com/Livox-SDK/livox_ros_driver.git /opt/fast-lio/src/livox_ros_driver \
    && git -C /opt/fast-lio/src/livox_ros_driver checkout --detach "${LIVOX_ROS_DRIVER_REF}" \
    && git clone https://github.com/hku-mars/FAST_LIO.git /opt/fast-lio/src/FAST_LIO \
    && git -C /opt/fast-lio/src/FAST_LIO checkout --detach "${FAST_LIO_REF}" \
    && git -C /opt/fast-lio/src/FAST_LIO submodule update --init --recursive

RUN source /opt/ros/noetic/setup.bash \
    && cd /opt/fast-lio \
    && catkin_make -DCMAKE_BUILD_TYPE=Release

WORKDIR /workspace
CMD ["bash"]
