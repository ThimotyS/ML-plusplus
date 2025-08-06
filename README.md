# ML-PlusPlus
GUI implementation of ML-Quadrat. The DSL is implemented using [Sirius-Web](https://eclipse.dev/sirius/sirius-web.html), the front-end and back-end to launch the code generators are developed with [Vue.js](https://vuejs.org/) and Spring Boot(https://spring.io/projects/spring-boot) respectively.

# Software Requirements
To run the project
- [Docker](https://www.docker.com/)
- [Maven](https://maven.apache.org/index.html)
- [npm](https://www.npmjs.com/)

# Optional:
To run code generators in offline mode:
- [Java11](https://www.java.com/nl/)
- [java21](https://www.java.com/nl/)

# Getting Started
## Clone the repository
```bash
  git clone https://github.com/ThimotyS/ML-plusplus
  cd ML-plusplus
```
## Create two empty directories within the project root
Fromt project root, do the following:
```bash
  mkdir backend-storage-files
  mkdir backend-temp-files
```
## Build the back-end
Fromt project root, do the following:
```bash
  cd ml-quadrat-backend
  mvn clean package
```

## Build the front-end
Fromt project root, do the following:
```bash
  cd ml-quadrat-web
  npm run build
```

## Run the project with docker
```bash
  docker-compose build
  docker-compose up
```
