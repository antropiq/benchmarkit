import { ref } from 'vue'

const STORAGE_KEY = 'benchmarkit:theme'

function currentTheme() {
  const attr = document.documentElement.getAttribute('data-theme')
  return attr === 'light' ? 'light' : 'dark'
}

const theme = ref(currentTheme())

function toggle() {
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
  document.documentElement.setAttribute('data-theme', theme.value)
  try {
    window.localStorage.setItem(STORAGE_KEY, theme.value)
  } catch {
    // Private mode / storage disabled: the toggle still works for the session.
  }
}

export function useTheme() {
  return { theme, toggle }
}
