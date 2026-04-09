/** @type {import('tailwindcss').Config} */
module.exports = {
  devServer: { disableHostCheck: true },
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
