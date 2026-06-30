// Body-font comparison toggle. Progressive enhancement only.
// Flips between the serif reading body and an Inter body so the
// author can decide which they prefer. Choice persists in localStorage.
(function () {
  var KEY = "blog-body-font";
  var body = document.body;
  var btnSerif = document.getElementById("ft-serif");
  var btnInter = document.getElementById("ft-inter");
  if (!btnSerif || !btnInter) return;

  function apply(font) {
    var inter = font === "inter";
    body.classList.toggle("font-inter", inter);
    btnInter.classList.toggle("active", inter);
    btnSerif.classList.toggle("active", !inter);
  }

  apply(localStorage.getItem(KEY) || "serif");

  btnSerif.addEventListener("click", function () {
    localStorage.setItem(KEY, "serif");
    apply("serif");
  });
  btnInter.addEventListener("click", function () {
    localStorage.setItem(KEY, "inter");
    apply("inter");
  });
})();
