package com.mlquadrat.ml_quadrat_backend.project;

import java.io.BufferedReader;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.apache.tomcat.util.http.fileupload.IOUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import com.mlquadrat.ml_quadrat_backend.config.StorageService;
import com.mlquadrat.ml_quadrat_backend.mluser.MLUser;
import com.mlquadrat.ml_quadrat_backend.mluser.MLUserService;
import response.MLQuadratError;

@Service
public class ProjectService {
	
	
	private final ProjectRepository repo;
	
	
	@Value("${execution.time.project}")
	private int projectExecutionTime;
	
	@Value("${execution.time.images}")
	private int imageExecutionTime;
	
	@Autowired
	MLUserService mlUserService;
	
	@Autowired
	StorageService storageService;
	
	public ProjectService(ProjectRepository repo) {
		this.repo = repo;
	}
	
	public List<Project> getProjects() {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		return repo.findAllByMLUser(user.getId());
	}

	public Optional<Project> getProject(Integer id) {
		MLUser user = this.mlUserService.getAuthenticatedUser();

		return repo.findByIdAndUser(id, user.getId());
	}
	
	private Path sirius_web_to_desktop(String originalProjectPath) throws Exception {
		Path scriptPath = this.storageService.getFile_converter_path();
		ProcessBuilder pB = new ProcessBuilder("java", "-jar", scriptPath.toString(),originalProjectPath);
		Process process = pB.start();

		BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
		StringBuilder output = new StringBuilder();
		String line1;
		
		int exitCode1 = process.waitFor();
		while ((line1 = reader.readLine()) != null) {
			output.append(line1).append(System.lineSeparator());
		}
		String result = output.toString().trim();
		
		return Paths.get(result);
	}
	private Path m2c(String convertedFilePath, Path destinationPath) throws MLQuadratError {
//		Path scriptPath = Paths.get(System.getProperty("user.dir") + "/scripts/m2c.jar");
		Path scriptPath = this.storageService.getFile_m2c_path();

		ProcessBuilder pB = new ProcessBuilder("java", "-jar", scriptPath.toString(), convertedFilePath, destinationPath.getParent().toString() +"/");
		Process process;
		try {
			process = pB.start();
		} catch (Exception ex) {
			throw new MLQuadratError("There's an error running the following process: " + pB.toString() + " - " + ex.getMessage());
		}
		BufferedReader reader;
		try {
			reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
		} catch (Exception ex){
			throw new MLQuadratError("There's regarding the reading of the output of the process: " + pB.toString() + " - " + ex.getMessage());
		}
		StringBuilder output = new StringBuilder();
		String line2;
		
		try {
			int exitCode2 = process.waitFor();
		} catch (InterruptedException ex) {
			throw new MLQuadratError("There's an error waiting for the process to finish: " + pB.toString() + " - " + ex.getMessage());
		}
		try {
			while ((line2 = reader.readLine()) != null) {
				output.append(line2).append(System.lineSeparator());
			}
		} catch (IOException ex) {
			throw new MLQuadratError("There's an error reading the output of the process: " + pB.toString() + " - " + ex.getMessage());
		}
		
		String result2 = output.toString().trim();
		if (result2 == null) {
			throw new MLQuadratError("Error performing m2c transoformation");
		}
		return Paths.get(result2);
	}
	public void addProject(MultipartFile file) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		
		Optional<Project> proj = repo.findByName(file.getOriginalFilename());
		if (proj.isPresent()) {
			throw new MLQuadratError("Project already exists");
		}
		Path destinationPath;
		try {
			destinationPath = this.storageService.storeOriginalFile(file);
		} catch (Exception ex) {
			// TODO Auto-generated catch block
			throw new MLQuadratError("Can't save original file into database: " + ex.getMessage());
		}
		Project newProject = new Project(null, file.getOriginalFilename(), null,null, user, destinationPath.toString(), null);
		
//		Path scriptPath1 = Paths.get(System.getProperty("user.dir") + "/scripts/sirius_web_to_desktop.jar");
//		ProcessBuilder pB1 = new ProcessBuilder("java", "-jar", scriptPath1.toString(), newProject.getOriginalFilePath());
//		Process process1 = pB1.start();
//
//		BufferedReader reader1 = new BufferedReader(new InputStreamReader(process1.getInputStream()));
//		StringBuilder output1 = new StringBuilder();
//		String line1;
//		
//		int exitCode1 = process1.waitFor();
//		while ((line1 = reader1.readLine()) != null) {
//			output1.append(line1).append(System.lineSeparator());
//		}
//		String result = output1.toString().trim();
//		
//		Path filePath1 = Paths.get(result);
		
		Path filePath1;
		try {
			filePath1 = this.sirius_web_to_desktop(newProject.getOriginalFilePath());
			newProject.setConvertedFileName(filePath1.getFileName().toString());
			newProject.setConvertedFilePath(filePath1.toString());
		} catch (Exception ex) {
			// TODO Auto-generated catch block
			throw new MLQuadratError("Can't convert sirius web .xml to EMF xml format: " + ex.getMessage());
		}

		Path filePath2;
		try {
			filePath2 = this.m2c(newProject.getConvertedFilePath(), destinationPath);
			newProject.setThingMLFileName(filePath2.getFileName().toString());
			newProject.setThingMLFilePath(this.storageService.storeLocation(filePath2.toString()).toString());
		} catch (MLQuadratError ex) {
			throw new MLQuadratError("Can't convert to .thingml file: " + ex.getMessage());
		}
		
		newProject.setUploadDate(new Date());
		repo.save(newProject);
	}
	
	public Resource downloadOriginal(Integer projectId) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);

		return this.storageService.loadFileByName(proj.getOriginalFileName());
	}
	
	
	public Resource downloadConverted(Integer projectId) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);

		return this.storageService.loadFileByPath(proj.getConvertedFilePath());
	}
	
	
	public Resource downloadThingML(Integer projectId) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);

		return this.storageService.loadFileByPath(proj.getThingMLFilePath());
	}
	
	//https://www.baeldung.com/spring-boot-requestmapping-serve-zip
	//https://stackoverflow.com/questions/57997257/how-can-i-zip-a-complete-directory-with-all-subfolders-in-java
	//check how to zip whole project structure
	public String downloadThingMLProject(Integer projectId, ZipOutputStream zipOutputStream) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);
		
		Path projPath = Paths.get(proj.getThingMLProjectPath());		
		Files.walkFileTree(projPath, new SimpleFileVisitor<Path>() {
		       @Override
		       public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
		    	   zipOutputStream.putNextEntry(new ZipEntry(projPath.relativize(file).toString()));
		           Files.copy(file, zipOutputStream);
		           zipOutputStream.closeEntry();
		           return FileVisitResult.CONTINUE;
		        }
		    });
		zipOutputStream.close();
		return proj.getThingMLProjectName();
	}
	
	public Resource downloadGeneratedOutput(Integer projectId) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);
		if(proj.getThingMLProjectOutputPath() == null) {
			throw new MLQuadratError("Project has not been run yet, can not retrieve generated output.");
		}
		return this.storageService.loadFileByPath(proj.getThingMLProjectOutputPath());
	}
	
	
	public Resource downloadGeneratedReport(Integer projectId) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);
		Path projVis = this.storageService.getProjectVisualisationPath(proj).resolve("html_report.html");
		System.out.println("GETTING PROJVIS");
		System.out.println(projVis.toString());
		if(!Files.exists(projVis)) {
			throw new MLQuadratError("Project has not been run yet, can not retrieve generated report.");
		}
		return this.storageService.loadFileByPath(projVis.toString());
	}
	
	public void deleteProject(Integer projectId) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);

		this.storageService.deleteProjectFiles(proj);
		this.repo.delete(proj);
	}
	
	public void generate(Integer projectId) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = null;
		try {
			proj = this.checkUserProject(user, projectId);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			throw e;
		}
		if (proj != null && proj.getThingMLProjectPath() != null && Paths.get(proj.getThingMLProjectPath()) != null && Files.exists(Paths.get(proj.getThingMLProjectPath())) && Files.isDirectory(Paths.get(proj.getThingMLProjectPath()))) {
	        Files.walk(Paths.get(proj.getThingMLProjectPath()))
	            .sorted(Comparator.reverseOrder())
	            .forEach(path -> {
	                try {
	                    Files.delete(path);
	                } catch (IOException e) {
	                    throw new RuntimeException("Failed to delete " + path, e);
	                }
            });
		}

		Path destinationPath = storageService.getDestionationPath(user);
		Path newDirPath = null;
		try {
			String withoutExtension = proj.getThingMLFileName().replaceFirst("\\.thingml$","");
			newDirPath = destinationPath.resolve(withoutExtension);
			Files.createDirectories(newDirPath);
		} catch (Exception e) {
			throw new MLQuadratError("Could not create project directory for project");
		}
		Path scriptPath = this.storageService.getFile_generator_path();
		Path javaPath = this.storageService.getJava11();
		ProcessBuilder pB = new ProcessBuilder(javaPath.toString(), "-jar" , scriptPath.toString(), "-c", "auto", "-s", proj.getThingMLFilePath(), "-o", newDirPath.toString());
		pB.redirectErrorStream(true);
		Process process = null;
		
		
		try {
			process = pB.start();
		} catch (IOException ex) {
			// TODO Auto-generated catch block
			throw new MLQuadratError("Error starting project generation: " + pB.toString() + " - " + ex.getMessage());
		}
		
		BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
		String line;
        List<String> errorLines = new ArrayList<>();
		try {
			while ((line = reader.readLine()) != null) {
                if (line.startsWith("ERROR:") || line.contains("Error in file")) {
                    errorLines.add(line);
                }
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		};
		
		if (!errorLines.isEmpty()) {
		    String errorMessage = "There are errors when generating the project:\n" + String.join("\n", errorLines);
		    throw new MLQuadratError(errorMessage);
		}
		
		try {
			int exitCode = process.waitFor();
		} catch (InterruptedException ex) {
			// TODO Auto-generated catch block
			throw new MLQuadratError("Error waiting for project generation to finish: " + ex.getMessage());
		}
		
		if(!Files.exists(newDirPath)) {
			throw new MLQuadratError("Error generating project");
		}
		
		proj.setThingMLProjectPath(newDirPath.toAbsolutePath().toString());
		proj.setThingMLProjectName(newDirPath.getFileName().toString());
		this.repo.save(proj);
	}
	
	public Project checkUserProject(MLUser user, Integer id) throws Exception {
		Optional<Project> optionalProj = repo.findById(id);
		
		if (!optionalProj.isPresent()) {
			throw new Exception("No Project found with this id");
		}
		Project proj = optionalProj.get();
		if(!proj.getMl_user().getId().equals(user.getId())) {
			throw new Exception("User is not the owner of the project");
		}
		return proj;
	}
	
	
	public void executeProject(Integer projectId) throws Exception {
		System.out.println("Executing Project");
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj;
		try {
			proj = this.checkUserProject(user, projectId);
		} catch (Exception e) {
			throw e;
		}
		
		if (proj.getThingMLProjectPath() == null) {
			throw new MLQuadratError("Cannot execute project, the project has not been generated yet");
		}
		
		
		Path scriptPath = Paths.get(proj.getThingMLProjectPath()).resolve("python_java/pom.xml");
		ProcessBuilder pB = new ProcessBuilder("mvn", "-f", scriptPath.toString(), "clean", "package", "-DskipTests");
		Process process;
		System.out.println("Packaging Project");
		System.out.println(pB.command().toString());

		try {
			process = pB.start();
		    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
		    String line;
		    while ((line = reader.readLine()) != null) {
		        System.out.println(line);
		    }
		} catch (Exception ex) {
			throw new MLQuadratError("Error starting project execution: " + pB.toString() + " - " + ex.getMessage());
		}
		process.waitFor();
		
		System.out.println("mkdir data and copying ");
		if(proj.getDatasetName() != null) {
				Path sourcePath = Paths.get(proj.getDatasetPath());
				Path targetDir = Paths.get(proj.getThingMLProjectPath() + "/python_java/target/data");
				Files.createDirectories(targetDir);
				Path targetPath = targetDir.resolve(sourcePath.getFileName());

				Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);		
		}
		Path dirPath = Paths.get(proj.getThingMLProjectPath() + this.storageService.getProjectExecution());
		Path jarPath = Files.list(dirPath)
	            .filter(p -> p.getFileName().toString().contains("with-dependencies"))
	            .filter(p -> p.getFileName().toString().endsWith(".jar"))
	            .findFirst()
	            .orElseThrow(() -> new RuntimeException("No JAR with 'with-dependencies' found in " + dirPath));		
		ProcessBuilder execute = new ProcessBuilder(this.storageService.getJava11().toString(), "-jar", jarPath.toString());
		
		execute.redirectErrorStream(true);
		execute.directory(new File(dirPath.toString()));
		
		Map<String, String> env = execute.environment();
		String condaBin = this.storageService.getCondaPath().toString();
		String existingPath = env.get("PATH");
		env.put("PATH", condaBin + ":" + existingPath);
		System.out.println("Running Project jar file");
		System.out.println(execute.command().toString());
		Process executeProcess;		

		System.out.println("Will stop the process after: " + this.projectExecutionTime + " seconds");
		try {
			executeProcess = execute.start();
			
			BufferedReader reader = new BufferedReader(new InputStreamReader(executeProcess.getInputStream()));
			StringBuilder output = new StringBuilder();

			ExecutorService executor = Executors.newSingleThreadExecutor();
			Future<?> future = executor.submit(() -> {
			    try {
			        String line;
			        while ((line = reader.readLine()) != null) {
			            output.append(line).append(System.lineSeparator());
			            System.out.println(line);
			        }
			    } catch (IOException e) {
			        e.printStackTrace();
			    }
			});
			
		    boolean finished = executeProcess.waitFor(this.projectExecutionTime, TimeUnit.SECONDS);
		    if (!finished) {
		        System.out.println("Execution Reached");
		        executeProcess.destroyForcibly();
		    }
			String filePath = this.storageService.getDestionationPath(user) + "/" + proj.getThingMLFileName().replace(".thingml", "-output.txt");
			try (FileWriter writer = new FileWriter(new File(filePath))) {
			    writer.write(output.toString());
			}catch (Exception ex) {
				throw new MLQuadratError("Error saving project output to file: " + ex.getMessage());
			}
			proj.setThingMLProjectOutputPath(filePath);

			this.repo.save(proj);
		} catch (Exception ex) {
			throw new MLQuadratError("Error starting project execution: " + pB.toString() + " - " + ex.getMessage());
		}
		
		// Save output to file
//		String filePath = this.storageService.getDestionationPath(user) + "/" + proj.getThingMLFileName().replace(".thingml", "-output.txt");
//		try (FileWriter writer = new FileWriter(new File(filePath))) {
//		    writer.write(output.toString());
//		}catch (Exception ex) {
//		ex.printStackTrace();
//	}
//		proj.setThingMLProjectOutputPath(filePath);
//		this.repo.save(proj);
	}
	
	public void generateImages(Integer projectId) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj;
		try {
			proj = this.checkUserProject(user, projectId);
		} catch (Exception e) {
			throw e;
		}
		//TODO CHANGE
		Path scriptPath = Paths.get(System.getProperty("user.dir") + "/src/main/resources/static/storage/loop.py");
		ProcessBuilder pB = new ProcessBuilder("python3", scriptPath.toString());
		Process process;
		try {
			process = pB.start();

		} catch (Exception ex) {
			throw new MLQuadratError("Error starting project execution: " + pB.toString() + " - " + ex.getMessage());
		}

        ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
        scheduler.schedule(() -> {
    		BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
    		StringBuilder output = new StringBuilder();
    		String line;
    		try {
				while ((line = reader.readLine()) != null) {
					output.append(line).append(System.lineSeparator());
				}
			} catch (Exception ex) {
    			ex.printStackTrace();
			};
    		File file = new File(this.storageService.getDestionationPath(user) + "/" + proj.getThingMLProjectName().replace(".thingml", "-output.txt"));
    		try (FileWriter writer = new FileWriter(file)) {
                writer.write(output.toString());
            } catch (Exception ex) {
    			ex.printStackTrace();
			}
            process.destroyForcibly();
            
        }, this.imageExecutionTime, TimeUnit.SECONDS);
        try {
			scheduler.awaitTermination(this.imageExecutionTime, TimeUnit.SECONDS);
		} catch (InterruptedException ex) {
			throw new MLQuadratError("Error executing project: " + pB.toString() + " - " + ex.getMessage());
		}
	}
	
	public void updateProject(Integer projectId, MultipartFile file) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);
		this.storageService.deleteProjectFiles(proj);
		Path destinationPath = this.storageService.storeOriginalFile(file);
		proj.setOriginalFileName(file.getOriginalFilename());
		proj.setOriginalFilePath(destinationPath.toString());
		Path filePath1 = this.sirius_web_to_desktop(proj.getOriginalFilePath());
		proj.setConvertedFileName(filePath1.getFileName().toString());
		proj.setConvertedFilePath(filePath1.toAbsolutePath().toString());
		Path filePath2 = this.m2c(proj.getConvertedFilePath(), destinationPath);
		proj.setThingMLFileName(filePath2.getFileName().toString());
		proj.setThingMLFilePath(destinationPath.getParent().resolve(filePath2).getFileName().toString().toString());
		this.repo.save(proj);
	}
	
	
	public String downloadImages(Integer projectId, ZipOutputStream zipOutputStream) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);
		
		Path projVis = this.storageService.getProjectVisualisationPath(proj);
		for(File file : new File(projVis.toString()).listFiles()) {
			zipOutputStream.putNextEntry(new ZipEntry(file.getName()));
			FileInputStream fileInputStream = new FileInputStream(file);
			IOUtils.copy(fileInputStream, zipOutputStream);
	        fileInputStream.close();
	        zipOutputStream.closeEntry();
	    }
	    zipOutputStream.finish();
	    zipOutputStream.flush();
	    IOUtils.closeQuietly(zipOutputStream);
	    return proj.getThingMLProjectName();
	}
	
	public Resource downloadDataset(Integer projectId) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);

		return this.storageService.loadFileByName(proj.getDatasetName());
	}
	
	
	public void addDataset(Integer projectId, MultipartFile file) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);
		Path destinationPath = this.storageService.storeOriginalFile(file);
		proj.setDatasetPath(destinationPath.toString());
		proj.setDatasetName(file.getOriginalFilename());
		this.repo.save(proj);
	}
	
	public void removeDataset(Integer projectId) throws Exception {
		MLUser user = this.mlUserService.getAuthenticatedUser();
		Project proj = this.checkUserProject(user, projectId);
		this.storageService.deleteFile(proj.getDatasetPath());
		proj.setDatasetPath(null);
		proj.setDatasetName(null);
		this.repo.save(proj);
	}
}
