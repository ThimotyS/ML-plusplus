<script setup>
    import {onMounted, ref, defineProps} from "vue";
    import {downloadDataset,uploadDataset, deleteDataset, generateImages, updateProject, deleteProject, downloadGeneratedOutput, downloadImages, downloadConvertedFile, downloadThingMLFile, downloadThingMLProject, getProject, generateThingMLProject, executeThingMLProject, downloadOriginalFile, downloadGeneratedReport } from '@/composables/ProjectRequests.ts';
    import {useRoute, useRouter} from 'vue-router';

    const route = useRoute();
    const router = useRouter();
    const project = ref(null);
    const file = ref(null);
    const dataset = ref(null)
    const downloadedImages = ref(null);
    const generatedOutput = ref(null);
    const generatedReport = ref(null);

    const textRef = ref(false)
    const imageRef = ref(false)
    const reportRef = ref(false)

    const handleProject = async () => {
        await getProject(route.params.id).then(data => { 
                project.value = data;
        });
    }

    const handleTextToggle = (event) => {
        textRef.value = !textRef.value;
    }

    const handleImageToggle = (event) => {
        imageRef.value = !imageRef.value;
    }

    const handleReportToggle = (event) => {
        reportRef.value = !reportRef.value;
    }

    const handleFileChange = (event) => {
        if ( event.target.files == null) file.value = null; 
        file.value = event.target.files[0];
    }
    
    const handleDatasetChange = (event) => {
        if (event.target.files == null) dataset.value = null; 
        dataset.value = event.target.files[0];
    }

    const handleGenerate = async () => {
        if (project.value == null) return;
        await generateThingMLProject(project.value.id);
        handleProject();
    }
    const handleM2c = async () => {
        if (project.value == null) return;
        await generateThingMLProject(project.value.id);
        handleProject();
    }

    const handleExecute = async () => {
        if (project.value == null) return;
        const response = await executeThingMLProject(project.value.id);
        if(response.ok){
            alert("Successful Execution, now retrieving outputs.");
            await downloadGeneratedOutput(project.value.id,false)
            .then(output =>{
                generatedOutput.value = output.split("\n");
            })
            .catch((err => {
                console.log(err);
            }));
            await downloadImages(project.value.id,false)
            .then(images =>{
                downloadedImages.value = images;
            })
            .catch((err => {
                console.log(err);
            }));
            await downloadGeneratedReport(project.value.id,false)
            .then(report => {
                generatedReport.value = report;
            })
        }
    }

    const handleDeleteAll = async () => {
        if (project.value == null) return;
        deleteProject(project.value.id).then(async () => {
            await router.push({name: 'projects'}).catch(() => {});}
        );
    }
    const handleDeleteDataset = async () => {
        if (project.value == null) return;
        deleteDataset(project.value.id)
        handleProject();
    }
    const handleUpdate = async () => {
        if (project.value == null || file.value == null) return;
        await updateProject(project.value.id,file.value);
        handleProject();
    }

    const handleDataset = async () => {
        if (project.value == null || dataset.value == null) return;
        await uploadDataset(project.value.id,dataset.value);
        handleProject();
    }

    const handleGeneratedOutput = async () => {
        if (project.value == null) return;
        downloadGeneratedOutput(project.value.id);
    }

    const handleGeneratedReport = async () => {
        if (project.value == null) return;
        downloadGeneratedReport(project.value.id);
    }

    const handleImages = async () => {
        if (project.value == null) return;
        downloadImages(project.value.id);
    }

    onMounted(async () => {
        handleProject();
    });
</script>

<template>
    <div v-if="project" :key="project">
        <div>
            <h1> 
                {{project.originalFileName}}
            </h1>
        </div>
        <div class="project-actions">
            <div class="multiple">
                <button class="cta main" @click="handleGenerate()">
                    Generate Project
                </button>
                <button class="cta danger" @click="handleDeleteAll()">
                    Delete All
                </button>
                <button class="cta danger" @click="handleDeleteDataset()">
                    Delete Dataset
                </button>
            </div>
            <div class="multiple">
                <div>
                    Update Project
                    <form @submit.prevent="handleUpdate()">
                        <input  @change="(event) => handleFileChange(event)"type="file" accept=".xml">
                        <button type="submit">Upload</button>
                    </form>
                </div>
                <div>
                    Upload Dataset
                    <form @submit.prevent="handleDataset()">
                        <input  @change="(event) => handleDatasetChange(event)"type="file" accept=".xls,.csv">
                        <button type="submit">Upload</button>
                    </form>
                </div>    
            </div>
        </div>
        <div class="project-info">
            <table>
                <thead>
                    <tr>
                        <th>
                            File Type
                        </th>
                        <th>
                            File Name
                        </th>
                        <th>
                            Status
                        </th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>
                            Original
                        </td>
                        <td>
                            <a v-if="project.originalFileName != null" @click="downloadOriginalFile(project.id)">{{ project.originalFileName }}</a>
                            <p v-else>Check Status</p>
                        </td>
                        <td>
                            <p v-if="project.originalFileName == null" >
                                Not Generated
                            </p>
                            <p v-else="">
                                Generated
                            </p>
                        </td>
                    </tr> 
                    <tr>
                        <td>
                            Converted
                        </td>
                        <td>
                            <a v-if="project.convertedFileName != null" @click="downloadConvertedFile(project.id)">{{ project.convertedFileName }}</a>
                            <p v-else>Check Status</p>
                        </td>
                        <td>
                            <p v-if="project.convertedFileName == null" >
                                Not Generated
                            </p>
                            <p v-else="">
                                Generated
                            </p>
                        </td>
                    </tr> 
                    <tr>
                        <td>
                            ThingML
                        </td>
                        <td>
                            <a v-if="project.thingMLFileName != null" @click="downloadThingMLFile(project.id)">{{ project.thingMLFileName }}</a>
                            <p v-else>Check Status</p>
                        </td>    
                        <td>
                            <p v-if="project.thingMLFileName == null" >
                                Not Generated
                            </p>
                            <p v-else="">
                                Generated
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td>
                            ThingML Project
                        </td>

                        <td>
                            <a v-if="project.thingMLProjectName != null" @click="downloadThingMLProject(project.id)" :key="project.thingMLProjectName">{{project.thingMLProjectName}}</a>
                            <p v-else>Check Status</p>
                        </td>
                        <td>
                            <p v-if="project.thingMLProjectName == null" :key="project.thingMLProjectName">
                                Not Generated
                            </p>
                            <p v-else="" >
                                Generated
                            </p>
                        </td>
                    </tr>

                    <tr>
                        <td>
                            Dataset
                        </td>

                        <td>
                            <a v-if="project.datasetName != null" @click="downloadDataset(project.id)" :key="project.datasetName">{{project.datasetName}}</a>
                            <p v-else>Check Status</p>
                        </td>
                        <td>
                            <p v-if="project.datasetName == null" :key="project.datasetName">
                                Not Uploaded
                            </p>
                            <p v-else="" >
                                Uploaded
                            </p>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        <div class="project-output" v-if="project.thingMLProjectName != null" :key="project.thingMLProjectName">
            <h2>Project Output</h2>
            <div class="project-actions">
                <div class="multiple">
                    <button class="cta main" @click="handleExecute()">
                        Execute Project
                    </button>
                    <!-- <button class="cta main" @click="handleGenerateImage()">
                        Generate Images
                    </button> -->
                    <button class="cta sub" @click="handleImages()">
                        Download Images
                    </button>
                    <button class="cta sub" @click="handleGeneratedOutput()">
                        Download Output
                    </button>

                    <button class="cta sub" @click="handleGeneratedReport()">
                        Download Report
                    </button>
                </div>
            </div>
            <div class="container">
                <div  class="output-button" @click="(event) => handleTextToggle(event)">
                    Generated Output
                </div>
                <div :class="['collapse',{open:textRef}]">
                    <div v-if="generatedOutput" class="text-content">
                        <p v-for="(line, index) in generatedOutput" :key="index" >{{ line }}</p>
                    </div>
                    <p v-else=""> There is no output, please execute the project</p>
                </div>
                <div class="output-button" @click="(event) => handleImageToggle(event)">
                    Generated Images
                </div>
                <div :class="['collapse',{open:imageRef}]">

                    <div v-if="downloadedImages" v-for="image in downloadedImages" :key="image">
                        <img  :src="image" style="width: 521px; height: 512px; object-fit: cover;"/>
                    </div>
                    <p v-else=""> There are no images, please execute the project</p>
                </div>

                <div class="output-button" @click="(event) => handleReportToggle(event)">
                    Generated Report
                </div>
                <div :class="['collapse',{open:reportRef}]">
                    <div v-if="generatedReport" class="text-content" >
                        <div v-html="generatedReport"></div>
                    </div>
                    <p v-else=""> There is no report, please execute the project</p>
                </div>
            </div>
        </div>
    </div>

</template>

<style scoped>

a {
    &:hover {
        color:grey;
        text-decoration: underline;
        cursor: pointer;
    }
}
button {
    &:hover {
        cursor: pointer;
        background-image: linear-gradient(rgb(0 0 0/10%) 0 0);
    }
}
.cta {
    font-size: 1.25rem;
    padding: 0.75rem;
    color: white;
    border: none;
    border-radius: 12px;
    &.main {
        background-color: #6c7ae0;
    }
    &.danger {
        background-color:#FF6961;
    }
    &.sub {
        background-color: #9ea7e8;
    }
}

.project-actions {
    div {
        margin: 1rem;
    }
    .multiple {
        display: flex;
        flex-direction: row;
        gap: 1rem;
        justify-content: center;
    }

}

.project-info{
    display: flex;
    flex-direction: column;
    justify-content: center;
    table {
        box-shadow: 0 0px 40px 0px rgba(0,0,0,0.15);
        th {
            
            color:white;
            background-color: #6c7ae0;
            padding:1rem;
        }
    }
}

.project-output {
    .container {
        display: flex;
        flex-direction: column;
    }
    p {
        margin:0;
    }
    img { 
        margin: 0.5rem;
    }
    .output-button {
        &:hover{
            background-color: lightgray;
        }
    }

    .collapse {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3 ease-in-out;
    }
    .collapse.open {
        max-height: 1024px;
        overflow: scroll;
    }
    .text-content {
        text-align: left;
    }
}

</style>