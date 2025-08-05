package com.mlquadrat.ml_quadrat_backend.mluser;



import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;

import com.fasterxml.jackson.annotation.JsonManagedReference;
import com.mlquadrat.ml_quadrat_backend.project.Project;
import java.util.ArrayList;
import java.util.List;

import org.springframework.web.bind.annotation.RequestMapping;

@Entity
@Table(name="ml_user")
public class MLUser {
	
	public MLUser(Integer id, String name, String password, List<Project> projects) {
		super();
		this.id = id;
		this.name = name;
		this.password = password;
		this.projects = projects;
	}

	@Id
	@GeneratedValue(strategy = GenerationType.AUTO)
	private Integer id;
	
	@Column(nullable = false, unique = true)
	private String name;
	
	@Column(nullable=false)
	private String password;
	
	@OneToMany(mappedBy="ml_user", cascade = CascadeType.ALL, orphanRemoval = true)
	@JsonManagedReference
	private List<Project> projects = new ArrayList<>();
	
	public MLUser() {
		
	}
	
	public Integer getId() {
		return id;
	}

	public void setId(Integer id) {
		this.id = id;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public String getPassword() {
		return password;
	}

	public void setPassword(String password) {
		this.password = password;
	}

	public List<Project> getProjects() {
		return projects;
	}

	public void setProjects(List<Project> projects) {
		this.projects = projects;
	}
	
	public void addProject(Project project) {
		this.projects.add(project);
	}
	
	public void removeProject(Project project) {
		this.projects.remove(project);
	}
	
}
