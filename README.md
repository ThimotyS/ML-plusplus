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
```

## Run the project with docker
From project root, run:
```bash
  docker-compose build
  docker-compose up
```

## Local development:
Script files are located in docker-full-stack/ml-quadrat-backend/src/main/resources/scripts
If you have .thingml files then you can use mlquadrat.jar to compile them.
If you have .xml files from Sirius Web, you first need to use sirius_web_to_desktop.jar using java21, then m2c.jar using java21 and mlquadrat.jar using java11
