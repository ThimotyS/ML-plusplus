package com.mlquadrat.ml_quadrat_backend.attachment;

import com.mlquadrat.ml_quadrat_backend.project.Project;

import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

@Entity
@Table(name="attachment")
public class Attachment {
	
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Integer id;
	
	private String name;
	
	@ManyToOne(fetch = FetchType.LAZY)
	private Project project;
	
	private String attachmentPath;
	
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

	public Project getProject() {
		return project;
	}

	public void setProject(Project project) {
		this.project = project;
	}

	public String getAttachment() {
		return attachmentPath;
	}

	public void setAttachment(String attachment) {
		this.attachmentPath = attachment;
	}
}
