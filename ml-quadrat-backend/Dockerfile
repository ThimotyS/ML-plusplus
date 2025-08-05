# Start from Ubuntu 
FROM ubuntu

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y wget curl bzip2 software-properties-common maven

# Add PPA for OpenJDK 21
RUN add-apt-repository ppa:openjdk-r/ppa -y && \
    apt-get update && \
    apt-get install -y openjdk-21-jdk openjdk-11-jdk

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create the conda env
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n ml2 python=3.8 && \
    conda install -y -n ml2 nb_conda jupyter numpy pandas matplotlib scikit-learn scikit-learn-intelex tensorflow-gpu gensim keras pytorch seaborn && \
    conda install -y -n ml2 jupyter_contrib_nbextensions -c conda-forge

# Copy your Spring Boot jar
ARG JAR_FILE=target/*.jar
COPY ./target/ml-quadrat-backend-0.0.1-SNAPSHOT.jar /app.jar

# Copy scripts and make executable
RUN mkdir -p /var/lib/scripts
COPY ./src/main/resources/scripts /var/lib/scripts
RUN chmod +x /var/lib/scripts/*

# By default, set JAVA_HOME to Java 21
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Expose port
EXPOSE 8080

# Run the Spring Boot app
#ENTRYPOINT ["java", "-jar", "/app.jar"]
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate ml2 && exec java -jar /app.jar"]
