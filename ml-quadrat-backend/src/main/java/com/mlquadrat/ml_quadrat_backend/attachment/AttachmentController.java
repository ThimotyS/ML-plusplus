package com.mlquadrat.ml_quadrat_backend.attachment;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.mlquadrat.ml_quadrat_backend.project.Project;

@RestController
@RequestMapping("/api/v1/attachments")
@CrossOrigin
public class AttachmentController {
	
	@Autowired
	AttachmentService service;
	
	@GetMapping("")
	public List<Attachment> getProjectAttachments(@RequestBody Project project){
		return service.getProjectAttachments(project);
	}
}
