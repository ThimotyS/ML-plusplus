package com.mlquadrat.ml_quadrat_backend.attachment;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface AttachmentRepository extends JpaRepository<Attachment, Integer> {
	
	@Query("SELECT a FROM Attachment a WHERE a.project.id = :project")
	List<Attachment> findAllByProject(@Param("project") Integer project);
	
}
