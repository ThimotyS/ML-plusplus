package com.mlquadrat.ml_quadrat_backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

import com.mlquadrat.ml_quadrat_backend.mluser.MLUser;
import com.mlquadrat.ml_quadrat_backend.mluser.MLUserRepository;
import com.mlquadrat.ml_quadrat_backend.project.Project;
import com.mlquadrat.ml_quadrat_backend.project.ProjectRepository;

import org.springframework.boot.CommandLineRunner;
@SpringBootApplication
public class MlQuadratBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(MlQuadratBackendApplication.class, args);
	}
	
	@Bean
	CommandLineRunner commandLineRunner(MLUserRepository repo) {
		
		return args -> {
			MLUser u1 = new MLUser(null, "test", "1234", null);
			repo.save(u1);
		};
	}
}
