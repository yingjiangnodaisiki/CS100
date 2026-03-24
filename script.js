const themeToggleButton = document.getElementById("theme-toggle");
const yearSpan = document.getElementById("year");
const storageKey = "personal_site_theme";

function applyTheme(theme) {
  if (theme === "light") {
    document.body.classList.add("light");
  } else {
    document.body.classList.remove("light");
  }
}

function initTheme() {
  const savedTheme = localStorage.getItem(storageKey);
  if (savedTheme === "light" || savedTheme === "dark") {
    applyTheme(savedTheme);
    return;
  }

  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  applyTheme(prefersLight ? "light" : "dark");
}

themeToggleButton?.addEventListener("click", () => {
  const isLight = document.body.classList.contains("light");
  const nextTheme = isLight ? "dark" : "light";
  applyTheme(nextTheme);
  localStorage.setItem(storageKey, nextTheme);
});

if (yearSpan) {
  yearSpan.textContent = String(new Date().getFullYear());
}

initTheme();
