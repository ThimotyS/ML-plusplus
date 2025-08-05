<script setup>
    import {onMounted, ref} from "vue";
    import {getProjects, uploadProject,deleteProject} from '@/composables/ProjectRequests.ts';
    import {useRouter} from 'vue-router';

    const projects = ref([]);

    const file = ref(null);

    const router = useRouter();



    const handleProjects = async () => {
        await getProjects().then(data => { 
            projects.value = data
;        });
    }

    const handleFileChange = (event) => {
        if ( event.target.files == null) file.value = null; 
        file.value = event.target.files[0];
    }

    const handleUpload = async () => {
        if (file.value == null) return;
        await uploadProject(file.value);
        handleProjects();
    }

    const handleDelete = async (project_id) => {
        await deleteProject(project_id)
        handleProjects();
    }

    const handleBrowse = async (event, projectId) => {
        await router.push({name: 'project', params:{id:projectId}}).catch(() => {});
    }

    onMounted(async () => {
        handleProjects();
    });

</script>

<template>
    <div class="project-list-container">

        <div>
            <p>Upload a new Project</p>
            <form @submit.prevent="handleUpload()">
                <input @change="(event) => handleFileChange(event)"type="file" accept=".xml">
                <button type="submit">Upload File</button>
            </form>
        </div>
        <div class="project-overview">
            <p>Project Overview</p>
            <table>
                <thead>
                    <tr>
                        <th>
                            Original File Name
                        </th>
                        <th>
                            Date
                        </th>
                        <th>
                            Inspect
                        </th>
                        <th>
                            Reupload
                        </th>
                        <th>
                            Delete
                        </th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="project in projects" :id="project">
                        <td>
                            <p>{{ project.originalFileName }}</p>
                        </td>
                        <td>
                            <p>{{ project.uploadDate }}</p>
                        </td>
                        <td>
                            <button @click="(event) => handleBrowse(event, project.id)">Inspect</button>
                        </td>
                        <td>
                            <input @change="(event) => handleFileChange(event)"type="file" accept=".thingml,.xml">
                            <button type="submit">Upload File</button>
                        </td>
                        <td>
                            <button @click="handleDelete(project.id)">x</button>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>

<style scoped>
.project-overview {
    display: flex;
    flex-direction: column;
    justify-content: center;
    table {
        border-radius: 10px;

        box-shadow: 0 0px 40px 0px rgba(0,0,0,0.15);
        th {
            color:white;
            background-color: #6c7ae0;
            padding:1rem;
        }
    }
}




</style>