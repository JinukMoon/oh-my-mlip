// Clicking the site name in the header goes to the home page, like the logo.
document.addEventListener("DOMContentLoaded", function () {
  var logo = document.querySelector(".md-header__button.md-logo");
  var title = document.querySelector(".md-header__title");
  if (!logo || !title) {
    return;
  }
  title.addEventListener("click", function () {
    window.location.href = logo.href;
  });
});
