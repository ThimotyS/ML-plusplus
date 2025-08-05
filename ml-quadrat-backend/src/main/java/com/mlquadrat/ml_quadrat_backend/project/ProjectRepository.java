package com.mlquadrat.ml_quadrat_backend.project;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;


public interface ProjectRepository extends JpaRepository<Project, Integer>{
	
	
	@Query("SELECT p FROM Project p WHERE p.ml_user.id = :user")
	List<Project> findAllByMLUser(@Param("user") Integer user);
	
	@Query("SELECT p FROM Project p WHERE p.id = :id AND p.ml_user.id = :user")
	Optional<Project> findByIdAndUser(@Param("id") Integer id, @Param("user") Integer user);
	
	@Query("SELECT p FROM Project p WHERE p.thingMLFileName = :name")
	Optional<Project> findByName(@Param("name") String name);
}
