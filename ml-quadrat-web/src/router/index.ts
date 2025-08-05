import { createRouter, createWebHistory } from 'vue-router'
import LoginView from '@/views/LoginView.vue'
import ProjectOverview from '@/views/projects/ProjectOverview.vue'
import ProjectList from '@/views/projects/ProjectList.vue'
import {loggedIn} from '@/composables/Authentication'
const routes = [
  {
    path: '/',
    name: 'login',
    component: LoginView
  },
  {
    path: '/projects',
    name: 'projects',
    component: ProjectList
  },
  {
    path: '/projects/:id',
    name: 'project',
    component: ProjectOverview
  },

  {
    path: '/:catchAll(.*)',
    name: 'NotFound',
    component: LoginView
  }

]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach(async (to, from) => {
  if (!loggedIn() && to.name != 'login'){
    return { name: 'login'}
  } else if( loggedIn() && to.name == 'login') {
    return { name: from.name}
  }
})
export default router
