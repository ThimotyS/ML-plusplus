package com.mlquadrat.ml_quadrat_backend.attachment;

import java.util.List;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.mlquadrat.ml_quadrat_backend.project.Project;
@Service
public class AttachmentService {
	
	@Autowired
	AttachmentRepository repo;
	
	public List<Attachment> getProjectAttachments(Project project) {
		return repo.findAllByProject(project.getId());
	}

}
