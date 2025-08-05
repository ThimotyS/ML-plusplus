# ML-PlusPlus
GUI implementation of ML-Quadrat. The DSL is implemented using [Sirius-Web](https://eclipse.dev/sirius/sirius-web.html), the front-end and back-end to launch the code generators are developed with [Vue.js](https://vuejs.org/) and Spring Boot(https://spring.io/projects/spring-boot) respectively.

# Software Requirements
- [Docker](https://www.docker.com/)
- [Java11](https://www.java.com/nl/)
- [Java21](https://www.java.com/nl/)


# Getting Started
## Clone the repository
```bash
  git clone https://github.com/ThimotyS/ML-plusplus
  cd ML-plusplus
```
## Create two empty directories within the project root
```bash
  mkdir backend-storage-files
  mkdir backend-temp-files
```
## Run the Project with Docker
```bash
  docker-compose build
  docker-compose up
```
