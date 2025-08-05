var BACKEND_API : string   = process.env.VUE_APP_BACKEND_API ? process.env.VUE_APP_BACKEND_API : "http://ERRORHOST:8080";

export async function login(form : MLUser) {

    try {
    const response = await fetch(`${BACKEND_API}/api/v1/user/login`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            name: form.name,
            password: form.password,
        })
    })
    const json = await response.json();
    localStorage.setItem("jwt", json.jwt);
    } catch (error){
        console.log(error);
    }
}



export function loggedIn(){
    return localStorage.getItem("jwt") !== null;
}
