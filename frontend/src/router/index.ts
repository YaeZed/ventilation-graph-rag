import { createRouter, createWebHistory } from 'vue-router'
import MainLayout from '@/layout/MainLayout.vue'
import HomeView from '@/views/HomeView.vue'
import LoginView from '@/views/LoginView.vue'
import RegisterView from '@/views/RegisterView.vue'
import SettingsView from '@/views/SettingsView.vue'
import StatsView from '@/views/StatsView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: LoginView,
    },
    {
      path: '/register',
      name: 'register',
      component: RegisterView,
    },
    {
      path: '/',
      component: MainLayout,
      children: [
        {
          path: '',
          redirect: '/chat',
        },
        {
          path: 'chat',
          name: 'chat',
          component: HomeView,
        },
        {
          path: 'chat/:conversationId',
          name: 'conversation',
          component: HomeView,
        },
        {
          path: 'stats',
          name: 'stats',
          component: StatsView,
        },
        {
          path: 'settings',
          name: 'settings',
          component: SettingsView,
        },
      ],
    },
  ],
})

export default router
