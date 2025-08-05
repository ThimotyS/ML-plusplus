package response;

import java.util.Date;

import com.mlquadrat.ml_quadrat_backend.project.Project;

public class ProjectDTO {

    private Integer id;
    private Date uploadDate;
    private String originalFileName;
    private String thingMLFileName;
    private String convertedFileName;
    private String thingMLProjectName;
    private String datasetName;

    public ProjectDTO(Project project) {
        this.id = project.getId();
        this.uploadDate = project.getUploadDate();
        this.originalFileName = project.getOriginalFileName();
        this.thingMLFileName = project.getThingMLFileName();
        this.convertedFileName = project.getConvertedFileName();
        this.thingMLProjectName = project.getThingMLProjectName();
        this.datasetName = project.getDatasetName();
    }
}
