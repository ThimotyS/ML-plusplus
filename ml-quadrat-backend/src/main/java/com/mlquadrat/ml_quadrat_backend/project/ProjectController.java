package com.mlquadrat.ml_quadrat_backend.project;

import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.net.http.HttpHeaders;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.zip.ZipOutputStream;

import org.apache.tomcat.util.http.fileupload.IOUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/v1/projects")
public class ProjectController {
	
	@Autowired
	private ProjectService service;
	
	
	@GetMapping("")
	public List<Project> getProjects() {
		return service.getProjects();
	}
	
	
	@GetMapping("/{id}")
	public Optional<Project> getProject(@PathVariable("id") Integer id) {
		return service.getProject(id);
	}
	
	
	@PostMapping("/{id}/generate")
	public ResponseEntity<Map<String,String>> generate(@PathVariable("id") Integer id) throws Exception{
		service.generate(id);
		Map<String,String> response = new HashMap();
		response.put("message", "Successfully generated project");
		return ResponseEntity.ok(response);
	}
	
	@PostMapping("/{id}/execute")
	public ResponseEntity<Map<String,String>> executeProject(@PathVariable("id") Integer id) throws Exception{
		service.executeProject(id);
		Map<String,String> response = new HashMap();
		response.put("message", "Successfully executed project");
		return ResponseEntity.ok(response);
	}
	
	@PostMapping("/{id}/generateImages")
	public ResponseEntity<Map<String,String>> generateImages(@PathVariable("id") Integer id) throws Exception{
		service.generateImages(id);
		Map<String,String> response = new HashMap();
		response.put("message", "Successfully generated images");
		return ResponseEntity.ok(response);
	}
	
	@GetMapping("/{id}/downloadImages")
	public ResponseEntity<byte[]> downloadImages(@PathVariable("id") Integer id) throws Exception{
		
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(byteArrayOutputStream);
        ZipOutputStream zipOutputStream = new ZipOutputStream(bufferedOutputStream);
        String projName = this.service.downloadImages(id, zipOutputStream);

		String headerValue = "attachment; filename=\"" + projName.replace(".thingml", ".zip") + "\"";

        IOUtils.closeQuietly(bufferedOutputStream);
        IOUtils.closeQuietly(byteArrayOutputStream);

        return ResponseEntity
          .ok()
          .header("Content-Disposition", headerValue)
          .body(byteArrayOutputStream.toByteArray());
	}
	
	@DeleteMapping("/{id}")
	public ResponseEntity<Map<String,String>> deleteProject(@PathVariable("id") Integer id) throws Exception{
		service.deleteProject(id);
		Map<String,String> response = new HashMap();
		response.put("message", "Successfully deleted project");
		return ResponseEntity.ok(response);	
	}
	
	
	@GetMapping("/{id}/downloadOriginal")
	public ResponseEntity<Resource> downloadOriginal(@PathVariable("id") Integer id) throws Exception{
		Resource res = service.downloadOriginal(id);
		String headerValue = "attachment; filename=\"" + res.getFilename() + "\"";
		
		return ResponseEntity.ok().contentLength(res.contentLength())
				.contentType(MediaType.APPLICATION_OCTET_STREAM)
				.header("Content-Disposition", headerValue)
				.body(res);
	}
	
	
	@GetMapping("/{id}/downloadConverted")
	public ResponseEntity<Resource> downloadConverted(@PathVariable("id") Integer id) throws Exception{
		
		Resource res = service.downloadConverted(id);
		String contentType = "application/octet-stream";
		String headerValue = "attachment; filename=\"" + res.getFilename() + "\"";
		
		return ResponseEntity.ok().contentType(MediaType.parseMediaType(contentType)).header("Content-Disposition", headerValue).body(res);
		
	}
	
	
	@GetMapping("/{id}/downloadThingML")
	public ResponseEntity<Resource> downloadThingML(@PathVariable("id") Integer id) throws Exception{
		Resource res = service.downloadThingML(id);
		String contentType = "application/octet-stream";
		String headerValue = "attachment; filename=\"" + res.getFilename() + "\"";
		
		return ResponseEntity.ok().contentType(MediaType.parseMediaType(contentType)).header("Content-Disposition", headerValue).body(res);
	}
	
	
	@GetMapping("/{id}/downloadGeneratedOutput")
	public ResponseEntity<Resource> downloadGeneratedOutput(@PathVariable("id") Integer id) throws Exception{
		
		Resource res = service.downloadGeneratedOutput(id);
		String contentType = "application/octet-stream";
		String headerValue = "attachment; filename=\"" + res.getFilename() + "\"";
		
		return ResponseEntity.ok().contentType(MediaType.parseMediaType(contentType)).header("Content-Disposition", headerValue).body(res);
		
	}
	
	@GetMapping("/{id}/downloadGeneratedReport")
	public ResponseEntity<Resource> downloadGeneratedReport(@PathVariable("id") Integer id) throws Exception{
		
		Resource res = service.downloadGeneratedReport(id);
		String contentType = "application/octet-stream";
		String headerValue = "attachment; filename=\"" + res.getFilename() + "\"";
		
		return ResponseEntity.ok().contentType(MediaType.parseMediaType(contentType)).header("Content-Disposition", headerValue).body(res);
	}
	
	@PutMapping(path="/{id}", consumes=MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Map<String,String>> updateProject(@PathVariable("id")Integer id, @RequestParam("file") MultipartFile file) throws Exception {
		service.updateProject(id, file);
		Map<String,String> response = new HashMap();
		response.put("message", "Successfully updated project");
		return ResponseEntity.ok(response);	
	}
	
	@PostMapping(path="", consumes=MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Map<String,String>> addProject(@RequestPart("file") MultipartFile file) throws Exception {		
		service.addProject(file);
		Map<String,String> response = new HashMap();
		response.put("message", "Successfully added project");

		return ResponseEntity.ok(response);
	}
	
	@GetMapping(path="/{id}/downloadThingMLProject", produces="application/zip")
	public ResponseEntity<byte[]> downloadThingMLProject(@PathVariable("id") Integer id) throws Exception {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(byteArrayOutputStream);
        ZipOutputStream zipOutputStream = new ZipOutputStream(bufferedOutputStream);
        String projName = this.service.downloadThingMLProject(id, zipOutputStream);

		String headerValue = "attachment; filename=\"" + projName.replace(".thingml", ".zip") + "\"";

        IOUtils.closeQuietly(bufferedOutputStream);
        IOUtils.closeQuietly(byteArrayOutputStream);

        return ResponseEntity
          .ok()
          .header("Content-Disposition", headerValue)
          .body(byteArrayOutputStream.toByteArray());
	}
	
	
	@PostMapping(path="/{id}/uploadDataset", consumes=MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Map<String,String>> uploadDataset(@PathVariable("id") Integer id,@RequestPart("file") MultipartFile file) throws Exception {		
		service.addDataset(id, file);
		Map<String,String> response = new HashMap();
		response.put("message", "Successfully added project");
		return ResponseEntity.ok(response);
	}
	
	@GetMapping("/{id}/downloadDataset")
	public ResponseEntity<Resource> downloadDataset(@PathVariable("id") Integer id) throws Exception{
		Resource res = service.downloadDataset(id);
		String contentType = "application/octet-stream";
		String headerValue = "attachment; filename=\"" + res.getFilename() + "\"";
		
		return ResponseEntity.ok().contentType(MediaType.parseMediaType(contentType)).header("Content-Disposition", headerValue).body(res);
	}
	
	@DeleteMapping("/{id}/deleteDataset")
	public ResponseEntity<Map<String,String>> deleteDataset(@PathVariable("id") Integer id) throws Exception{
		service.removeDataset(id);
		Map<String,String> response = new HashMap();
		response.put("message", "Successfully deleted project");
		return ResponseEntity.ok(response);	
	}
}

