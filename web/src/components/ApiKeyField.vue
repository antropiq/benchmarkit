<script setup>
import { ref } from 'vue'

defineProps({
  modelValue: { type: String, default: '' },
  placeholder: { type: String, default: 'not-needed' },
})

const emit = defineEmits(['update:modelValue', 'change'])

const visible = ref(false)
</script>

<template>
  <span class="password-field">
    <input
      :value="modelValue"
      :type="visible ? 'text' : 'password'"
      :placeholder="placeholder"
      autocomplete="off"
      spellcheck="false"
      @input="emit('update:modelValue', $event.target.value)"
      @change="emit('change', $event)"
    />
    <button
      class="password-toggle"
      type="button"
      :aria-label="visible ? 'Hide API key' : 'Show API key'"
      :aria-pressed="visible"
      @click.prevent="visible = !visible"
    >
      <svg
        v-if="visible"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
        <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
        <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
        <line x1="1" y1="1" x2="23" y2="23" />
      </svg>
      <svg
        v-else
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    </button>
  </span>
</template>
