package com.mlquadrat.ml_quadrat_backend.project;


import jakarta.persistence.CascadeType;	

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;

import com.fasterxml.jackson.annotation.JsonBackReference;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.mlquadrat.ml_quadrat_backend.attachment.Attachment;
import com.mlquadrat.ml_quadrat_backend.mluser.MLUser;

import java.util.Date;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name="project")
public class Project {

	
	@Id
	@GeneratedValue(strategy = GenerationType.AUTO)
	private Integer id;
	
	@Column(name="upload_date", nullable=false)
    @JsonFormat( pattern = "dd-MM-yyyy")
	private Date uploadDate;
	
	@Column(name="original_file_name",nullable = false, unique = true)
	private String originalFileName;
	
	@Column(name="thingml_file", unique = true)
	private String thingMLFileName;
	
	@Column(name="converted_file_name", unique = true)
	private String convertedFileName;
	
	@Column(name="thingml_project_name", unique=true)
	private String thingMLProjectName;
	
	
	@OneToMany(mappedBy="project", cascade = CascadeType.ALL, orphanRemoval = true)
	private List<Attachment> attachments = new ArrayList<>();
	
	@ManyToOne(fetch = FetchType.LAZY)
	@JoinColumn(name="p_id", referencedColumnName="id")
	@JsonBackReference
	private MLUser ml_user;
	
	
	@Column(name="original_file_path", unique=true)
	private String originalFilePath;
	
	@Column(name="converted_file_path", unique=true)
	private String convertedFilePath;
	
	@Column(name="thingml_file_path")
	private String thingMLFilePath;

	@Column(name="thingml_project_path", unique=true)
	private String thingMLProjectPath;
	
	@Column(name="dataset_name", unique=true)
	private String datasetName;
	
	@Column(name="dataset_path", unique=true)
	private String datasetPath;


	@Column(name="thingml_project_output_path", unique=true)
	private String thingMLProjectOutputPath;
	
	public String getConvertedFileName() {
		return convertedFileName;
	}

	public void setConvertedFileName(String convertedFileName) {
		this.convertedFileName = convertedFileName;
	}

	public String getConvertedFilePath() {
		return convertedFilePath;
	}

	public void setConvertedFilePath(String convertedFilePath) {
		this.convertedFilePath = convertedFilePath;
	}


	public Integer getId() {
		return id;
	}

	public void setId(Integer id) {
		this.id = id;
	}

	public String getOriginalFileName() {
		return originalFileName;
	}

	public void setOriginalFileName(String originalFileName) {
		this.originalFileName = originalFileName;
	}

	public String getThingMLFileName() {
		return thingMLFileName;
	}

	public void setThingMLFileName(String thingMLFileName) {
		this.thingMLFileName = thingMLFileName;
	}

	public List<Attachment> getAttachments() {
		return attachments;
	}

	public void setAttachments(List<Attachment> attachments) {
		this.attachments = attachments;
	}

	public MLUser getMl_user() {
		return ml_user;
	}

	public void setMl_user(MLUser ml_user) {
		this.ml_user = ml_user;
	}

	public String getOriginalFilePath() {
		return originalFilePath;
	}

	public void setOriginalFilePath(String originalFile) {
		this.originalFilePath = originalFile;
	}

	public String getThingMLFilePath() {
		return thingMLFilePath;
	}

	public void setThingMLFilePath(String thingMLFile) {
		this.thingMLFilePath = thingMLFile;
	}
	public String getThingMLProjectPath() {
		return thingMLProjectPath;
	}

	public void setThingMLProjectPath(String thingml_project_path) {
		this.thingMLProjectPath = thingml_project_path;
	}
	
	public String getThingMLProjectName() {
		return thingMLProjectName;
	}

	public void setThingMLProjectName(String thingMLProjectName) {
		this.thingMLProjectName = thingMLProjectName;
	}
	
	public String getThingMLProjectOutputPath() {
		return thingMLProjectOutputPath;
	}

	public void setThingMLProjectOutputPath(String thingMLProjectOutputPath) {
		this.thingMLProjectOutputPath = thingMLProjectOutputPath;
	}
	
	public Project() {
		
	}

	public Project(Integer id, String originalFileName, String thingMLFileName, List<Attachment> attachments,
			MLUser ml_user, String originalFile, String thingMLFile) {
		super();
		this.id = id;
		this.originalFileName = originalFileName;
		this.thingMLFileName = thingMLFileName;
		this.attachments = attachments;
		this.ml_user = ml_user;
		this.originalFilePath = originalFile;
		this.thingMLFilePath = thingMLFile;
	}

	public Date getUploadDate() {
		return uploadDate;
	}

	public void setUploadDate(Date uploadDate) {
		this.uploadDate = uploadDate;
	}

	public String getDatasetPath() {
		return datasetPath;
	}

	public void setDatasetPath(String datasetPath) {
		this.datasetPath = datasetPath;
	}

	public String getDatasetName() {
		return datasetName;
	}

	public void setDatasetName(String datasetName) {
		this.datasetName = datasetName;
	}
	
	
}
