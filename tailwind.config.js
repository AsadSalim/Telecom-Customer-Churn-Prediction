/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./Form.html'],
  theme: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
