package com.mlquadrat.ml_quadrat_backend.config;


import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import com.mlquadrat.ml_quadrat_backend.mluser.MLUser;
import com.mlquadrat.ml_quadrat_backend.mluser.MLUserRepository;
import com.mlquadrat.ml_quadrat_backend.mluser.MLUserService;
import com.mlquadrat.ml_quadrat_backend.project.Project;

import response.MLQuadratError;

@Service
public class StorageService {
	
	
	private final Path rootLocation;
	
	private final Path tempLocation;
	
	private final String projectVisualisation;
	
	private final String projectExecution;
	
	private final Path file_converter_path;
	
	private final Path file_m2c_path;
	
	private final Path file_generator_path;
	
	private final Path java11;
	
	private final Path java21;
	
	private final Path condaPath;
	
	@Autowired
	MLUserRepository userRepo;
	
	@Autowired
	MLUserService mlUserService;
	
	
	StorageService(@Value("${file.java11.path}")String java11,@Value("${file.java21.path}")String java21, @Value("${file.scripts.path}")String scriptsPath,  @Value("${file.storage.path}")String storagePath, @Value("${file.temp.path") String tempPath, @Value("${file.project.visualisation}") String projectVisualisation,@Value("${file.project.execution}") String projectExecution,
			@Value("${file.conda.path}") String condaPath,
			MLUserRepository userRepo){
		this.rootLocation = Paths.get(storagePath);
		this.tempLocation = Paths.get(tempPath);
		this.projectVisualisation = projectVisualisation;
		this.projectExecution = projectExecution;
		this.file_converter_path = Paths.get(scriptsPath + "/sirius_web_to_desktop.jar");
		this.file_m2c_path = Paths.get(scriptsPath + "/m2c.jar");
		this.file_generator_path = Paths.get(scriptsPath + "/mlquadrat.jar");
		this.java11 = Paths.get(java11);
		this.java21 = Paths.get(java21);
		this.condaPath = Paths.get(condaPath);
		try {
			Files.createDirectories(rootLocation);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public Path getRootLocation() {
		return rootLocation;
	}

	
	public Path storeOriginalFile(MultipartFile file) throws Exception {
		Path desinationFile = storeLocation(file.getOriginalFilename());
		if(!desinationFile.getParent().getParent().equals(this.rootLocation.toAbsolutePath())) {
			throw new Exception("Cannot store file outside current directory");
		}
		try (InputStream inputStream = file.getInputStream()){
			Files.copy(inputStream, desinationFile, StandardCopyOption.REPLACE_EXISTING);
		}
		return desinationFile;
	}
	
	public Path storeLocation(String filename) throws Exception {
		Path path = null;
		try {
			path = this.getUserDir().resolve(filename).normalize().toAbsolutePath();
		} catch(Exception ex) {
			throw new MLQuadratError("Could not find Filename");
		}
		return path;
	}
	
	public Path getUserDir() throws IOException {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		return Files.createDirectories(this.rootLocation.resolve(user.getName()));
	}

	public Path loadLocation(String filename) throws IOException {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		return this.rootLocation.resolve(user.getName()).resolve(filename);
	}
	
	
	public Resource loadFileByName(String filename) throws Exception {
		Path file = loadLocation(filename);
		Resource res = new UrlResource(file.toUri());
		if (res.exists() || res.isReadable()) {
			return res;
		} else {
			throw new Exception("Could not read file");
		}
	}
	
	public Resource loadFileByPath(String path) throws Exception {
		Path file = Paths.get(path);
		Resource res = new UrlResource(file.toUri());
		if (res.exists() || res.isReadable()) {
			return res;
		} else {
			throw new Exception("Could not read file");
		}
	}
	
	public void deleteProjectFiles(Project proj) throws Exception  {
//		if(proj.getOriginalFilePath() != null) Files.deleteIfExists(Paths.get(proj.getOriginalFilePath()));
//		if(proj.getConvertedFilePath() != null) Files.deleteIfExists(Paths.get(proj.getConvertedFilePath()));
//		if(proj.getThingMLFilePath() != null) Files.deleteIfExists(Paths.get(proj.getThingMLFilePath()));
//		if(proj.getThingMLProjectOutputPath() != null) Files.deleteIfExists(Paths.get(proj.getThingMLProjectOutputPath()));
		this.deleteFile(proj.getOriginalFilePath());
		proj.setOriginalFileName(null);
		proj.setOriginalFilePath(null);
		this.deleteFile(proj.getConvertedFilePath());
		proj.setConvertedFileName(null);
		proj.setConvertedFilePath(null);
		this.deleteFile(proj.getThingMLFilePath());
		proj.setThingMLFileName(null);
		proj.setThingMLFilePath(null);
		this.deleteDir(proj.getThingMLProjectPath());
		proj.setThingMLProjectName(null);
		proj.setThingMLProjectPath(null);
		this.deleteFile(proj.getThingMLProjectOutputPath());
		proj.setThingMLProjectOutputPath(null);
		proj.setThingMLProjectOutputPath(null);
		this.deleteFile(proj.getDatasetPath());
		proj.setDatasetName(null);
		proj.setDatasetPath(null);
	}
	
	public void deleteFile(String fileName) throws Exception {
		if(fileName != null && !fileName.equals(this.getUserDir().toAbsolutePath().toString())) {
			Files.deleteIfExists(Paths.get(fileName));
		}
	}
	// Generated by chat gpt
	public void deleteDir(String dirName) throws Exception {
		Path path = Paths.get(dirName);
        if (Files.exists(path)) {
            Files.walk(path)
                .sorted(Comparator.reverseOrder())
                .forEach(p -> {
                    try {
                        Files.delete(p);
                    } catch (IOException e) {
                        throw new RuntimeException("Failed to delete " + p, e);
                    }
            });
        }
	}
	
	
	public Path getDestionationPath(MLUser user) {
		return Paths.get(this.rootLocation.toAbsolutePath().toString() + "/" + user.getName() + "/");
	}
	
	public Path getProjectVisualisationPath(Project proj) {
		return Paths.get(proj.getThingMLProjectPath() + "/" + this.projectVisualisation);
	}

	public Path getProjectExecution(Project proj) {
		return Paths.get(proj.getThingMLProjectPath() + "/" + this.projectExecution);
	}

	public Path getFile_converter_path() {
		return file_converter_path;
	}

	public Path getFile_m2c_path() {
		return file_m2c_path;
	}

	public Path getFile_generator_path() {
		return file_generator_path;
	}

	public Path getJava11() {
		return java11;
	}

	public Path getJava21() {
		return java21;
	}

	public Path getCondaPath() {
		return condaPath;
	}

	public String getProjectVisualisation() {
		return projectVisualisation;
	}

	public String getProjectExecution() {
		return projectExecution;
	}
	
}
