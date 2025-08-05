//http://localhost:8080/api/v1/projects

import JSZip from "jszip";

var BACKEND_API : string   = process.env.VUE_APP_BACKEND_API ? process.env.VUE_APP_BACKEND_API : "http://ERRORHOST:8080";
// fetch(BACKEND_API + `/api/v1/projects/${project_id}`, {
//     method: 'GET',
//     headers: {
//         Authorization: `Bearer ${jwtToken}`
//     }
// }).then(response => {
//     if(!response.ok) { throw response}
//     return response.json()
// }).then(json => {
//     return json;
// }).catch(err => {
//     err.text().then( (errorMessage: any)=> {
//         alert(errorMessage);
//     })
// })


async function fetchWrapper(url:string, options={}){
    return await fetch(url,options)
    .then(response => {
        if(!response.ok) { 
            throw response
        }
        return response;
    }).catch(err => {
        err.json().then((errorMessage:any) => {
            alert(errorMessage.message);
        })
    })
}



export async function getProjects() {

    const jwtToken : string | null = localStorage.getItem("jwt");
    const response = await fetch(BACKEND_API + '/api/v1/projects', {
        method: 'GET',
        headers: {
            Authorization: `Bearer ${jwtToken}`
        }
    }).then(response => {
        if(!response.ok) { 
            localStorage.removeItem("jwt");
            throw response
        }
        return response.json()
    }).then(json => {
        return json;
    }).catch(err => {
        if (err.text){
        err.text().then( (errorMessage: any)=> {
            alert(errorMessage);
        })}
    })

    return response;
}

export async function getProject(project_id: number){
    try {
        const jwtToken : string | null = localStorage.getItem("jwt");
        const response = await fetch(BACKEND_API + `/api/v1/projects/${project_id}`, {
            method: 'GET',
            headers: {
                Authorization: `Bearer ${jwtToken}`
            }
        }).then(response => {
            if(!response.ok) { 
                localStorage.removeItem("jwt");
                throw response
            }
            return response.json()
        }).then(json => {
            return json;
        }).catch(err => {
            if (err.text){
                err.text().then( (errorMessage: any)=> {
                    alert(errorMessage);
                })}
        })
    
        return response;
    } catch (error){
        alert(error);
    }
}

export async function deleteProject( project_id: number) {
    
    const jwtToken : string | null = localStorage.getItem("jwt");
    return await fetchWrapper(BACKEND_API+`/api/v1/projects/${project_id}`, {
        method: 'DELETE',
        headers: {
            Authorization: `Bearer ${jwtToken}`
        }
    })
}



export async function uploadProject(file : File){
    const jwtToken : string | null = localStorage.getItem("jwt");
    const formData = new FormData();
    formData.append('file', file, file.name);
    return await fetchWrapper(BACKEND_API + '/api/v1/projects', {
        method: 'POST',
        body: formData,
        headers: {
            'Authorization': `Bearer ${jwtToken}`,
        },
    })
}


export async function updateProject(projectId: number, file : File){
    const jwtToken : string | null = localStorage.getItem("jwt");
    const formData = new FormData();
    formData.append('file', file, file.name);
    return await fetchWrapper(BACKEND_API + `/api/v1/projects/${projectId}`, {
        method: 'PUT',
        body: formData,
        headers: {
            'Authorization': `Bearer ${jwtToken}`,
        },
    })
}

function headerFileNameExtract(response :Response, fn: string){
    const disposition = response.headers.get('Content-Disposition')
    let filename = fn;
    if (disposition && disposition.includes("filename=")){
        const match = disposition.match(/filename="?([^"]+)"?/);
        if (match?.[1]) filename = match[1];
    }
    return filename;
}

//`http://localhost:8080/api/v1/projects/${project_id}/downloadOriginal`
export async function downloadOriginalFile(project_id: number){
    try {
        const jwtToken : string | null = localStorage.getItem("jwt");
        const response = await fetch(BACKEND_API +`/api/v1/projects/${project_id}/downloadOriginal`, {
            method: 'GET',
            headers: {
                Authorization: `Bearer ${jwtToken}`
            }
        })
        if (!response.ok) {
            const err = await response.json();
            alert(err.message);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = headerFileNameExtract(response,"original-file.xml"); 
        link.click();
        URL.revokeObjectURL(url);
    } catch (error){
        alert(error)
    }
}


export async function downloadConvertedFile(project_id: number){
    try {
        const jwtToken : string | null = localStorage.getItem("jwt");
        const response = await fetch(BACKEND_API + `/api/v1/projects/${project_id}/downloadConverted`, {
            method: 'GET',
            headers: {
                Authorization: `Bearer ${jwtToken}`
            }
        })
        if (!response.ok) {
            const err = await response.json();
            alert(err.message);      
          }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = headerFileNameExtract(response,"converted-file.thingml"); 
        link.click();
        URL.revokeObjectURL(url);
    } catch (error){
        alert(error)
    }
}


export async function downloadThingMLFile(project_id: number){
    try {
        const jwtToken : string | null = localStorage.getItem("jwt");
        const response = await fetch(BACKEND_API + `/api/v1/projects/${project_id}/downloadThingML`, {
            method: 'GET',
            headers: {
                Authorization: `Bearer ${jwtToken}`
            }
        })

        if (!response.ok) {
            const err = await response.json();
            alert(err.message);
        }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = headerFileNameExtract(response,"thingml-file.thingml"); 
        link.click();
        URL.revokeObjectURL(url);
    } catch (error){
        alert(error)
    }
}

export async function downloadThingMLProject(project_id: number){
    try {
        const jwtToken : string | null = localStorage.getItem("jwt");
        const response = await fetch(BACKEND_API + `/api/v1/projects/${project_id}/downloadThingMLProject`, {
            method: 'GET',
            headers: {
                Authorization: `Bearer ${jwtToken}`
            }
        })
        if (!response.ok) {
            const err = await response.json();
            alert(err.message);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = headerFileNameExtract(response,"project.zip"); 
        link.click();
        URL.revokeObjectURL(url);
    } catch (error){
        alert(error)
    }
}

//localhost:8080/api/v1/projects/3/generate
//`localhost:8080/api/v1/projects/${project_id}/generate`
export async function generateThingMLProject(project_id: number){
    const jwtToken : string | null = localStorage.getItem("jwt");
    return await fetchWrapper(BACKEND_API + `/api/v1/projects/${project_id}/generate`, {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${jwtToken}`
        }
    })
}

export async function executeThingMLProject(project_id: number) {
    const jwtToken : string | null = localStorage.getItem("jwt");
    return await fetchWrapper(BACKEND_API + `/api/v1/projects/${project_id}/execute`, {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${jwtToken}`
        }
    });
}

export async function generateImages(project_id: number) {
    const jwtToken : string | null = localStorage.getItem("jwt");
    return await fetchWrapper(BACKEND_API + `/api/v1/projects/${project_id}/generateImages`, {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${jwtToken}`
        }
    });
}

export async function downloadGeneratedOutput(project_id: number, download: boolean = true) {
    try {
        const jwtToken : string | null = localStorage.getItem("jwt");
        const response = await fetch(BACKEND_API + `/api/v1/projects/${project_id}/downloadGeneratedOutput`, {
            method: 'GET',
            headers: {
                Authorization: `Bearer ${jwtToken}`
            }
        })
        if (!response.ok) {
            const err = await response.json();
            alert(err.message);
            return;
        }

        const blob = await response.blob();
        if(download) {
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = headerFileNameExtract(response,"project.txt"); 
            link.click();
            URL.revokeObjectURL(url);
        } else {
            return blob.text();
        }
    } catch (error){
        alert(error)
    }
}

export async function downloadImages(project_id: number, download: boolean = true) {
    try {
        const jwtToken : string | null = localStorage.getItem("jwt");
        const response = await fetch(BACKEND_API + `/api/v1/projects/${project_id}/downloadImages`, {
            method: 'GET',
            headers: {
                Authorization: `Bearer ${jwtToken}`
            }
        })
        if (!response.ok) {
            const err = await response.json();
            alert(err.message);
            return;
        }

        const blob = await response.blob();
        if(download) {
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = headerFileNameExtract(response,"images.zip"); 
            link.click();
            URL.revokeObjectURL(url);
        } else {
            return await unzipFile(blob);
        }

    } catch (error){
        alert(error)
    }
}


export async function downloadGeneratedReport(project_id: number, download: boolean = true) {
    try {
        const jwtToken : string | null = localStorage.getItem("jwt");
        const response = await fetch(BACKEND_API + `/api/v1/projects/${project_id}/downloadGeneratedReport`, {
            method: 'GET',
            headers: {
                Authorization: `Bearer ${jwtToken}`
            }
        })
        if (!response.ok) {
            const err = await response.json();
            alert(err.message);
            return;
        }

        const blob = await response.blob();
        if(download) {
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = headerFileNameExtract(response,"report.html"); 
            link.click();
            URL.revokeObjectURL(url);
        } else {
            return blob.text();
        }
    } catch (error){
        alert(error)
    }
}


export async function unzipFile(data:any) {
    try {
        const zip = await JSZip.loadAsync(data)
    
        const urls: string[] = []
    
        for (const fileName in zip.files) {
          const file = zip.files[fileName]
          if (!file.dir && /\.(png|jpe?g|gif|webp)$/i.test(file.name)) {
            const blob = await file.async('blob')
            const url = URL.createObjectURL(blob)
            urls.push(url)
          }
        }
        return urls
      } catch (error) {
        console.error('Error loading or unpacking zip:', error)
      }
 }

 export async function downloadDataset(project_id: number){
    try {
        const jwtToken : string | null = localStorage.getItem("jwt");
        const response = await fetch(BACKEND_API +`/api/v1/projects/${project_id}/downloadDataset`, {
            method: 'GET',
            headers: {
                Authorization: `Bearer ${jwtToken}`
            }
        })
        if (!response.ok) {
            const err = await response.json();
            alert(err.message);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = headerFileNameExtract(response,"dataset.xls"); 
        link.click();
        URL.revokeObjectURL(url);
    } catch (error){
        alert(error)
    }
}
export async function uploadDataset(project_id:number, file : File){
    const jwtToken : string | null = localStorage.getItem("jwt");
    const formData = new FormData();
    formData.append('file', file, file.name);
    return await fetchWrapper(BACKEND_API + `/api/v1/projects/${project_id}/uploadDataset`, {
        method: 'POST',
        body: formData,
        headers: {
            'Authorization': `Bearer ${jwtToken}`,
        },
    })
}

export async function deleteDataset(project_id:number){
    const jwtToken : string | null = localStorage.getItem("jwt");
    return await fetchWrapper(BACKEND_API + `/api/v1/projects/${project_id}/deleteDataset`, {
        method: 'DELETE',
        headers: {
            'Authorization': `Bearer ${jwtToken}`,
        },
    })
}
