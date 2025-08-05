<script setup>

import {ref} from "vue";
import {login} from "../composables/Authentication.ts"
import {useRouter} from 'vue-router';
import {MLUser} from '@/models/MLUser';

const router = useRouter();


const loginForm = ref({
    name: '',
    password: ''
})

const handleLogin = async () => {
    await login(loginForm.value);
    await router.push({name: 'projects'}).catch(() => {});
}


</script>

<template>
     <div class="login-container">
                <div class="title">
                    ML-Quadrat Login
                </div>
                <form class="login-content" @submit.prevent="handleLogin()" >
                    <div>
                        <p>Login</p>
                        <input 
                            placeholder='Type in your login' 
                            v-model="loginForm.name"
                            required
                        />
                    </div>
                    <div>
                        <p>Password</p>
                        <input 
                            type="password" 
                            placeholder='Type in your Password' 
                            v-model="loginForm.password"
                            required
                        />
                    </div>

                    <div class="login-actions">
                        <button type="submit" >Login</button>
                    </div>
                </form>
            </div>
</template>

<style scoped>
.login-container{
    display:grid;
    max-width: 500px;
    border: 1px solid #6c7ae0;
    justify-content: center;
    margin:auto;
    .title {
        text-align: center;
        border-bottom: 1px solid #6c7ae0;
        width:100%
    }
}

.login-content{
    justify-self: center;
    display:flex;
    flex-direction: column;
    width: 20rem;
    div {
        display: flex;
        flex-direction: row;
        height: 3rem;
        p {
            width:5rem;
        }
        input {
            height: 1rem;
            align-self: center;
        }
    }
}

.login-actions {
    display:flex;
    justify-content: center;
    margin-bottom: 1rem;

    button {
        color: white;
        width:10rem;
        background-color: #6c7ae0;
        border-radius: 1rem;
        &:hover{
            background-image: linear-gradient(rgb(0 0 0/10%) 0 0);
            cursor:pointer;
        }
    }
}
</style>