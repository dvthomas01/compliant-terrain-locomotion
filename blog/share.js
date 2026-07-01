// Wire the share buttons to the live page URL and title, so they work wherever
// the post is hosted with no hardcoded canonical URL. Progressive enhancement:
// the anchors carry bare fallback hrefs, this fills in the real target on load.
(function () {
  var url = encodeURIComponent(window.location.href);
  var title = encodeURIComponent(document.title);
  var targets = {
    "share-x": "https://twitter.com/intent/tweet?url=" + url + "&text=" + title,
    "share-linkedin": "https://www.linkedin.com/sharing/share-offsite/?url=" + url,
    "share-facebook": "https://www.facebook.com/sharer/sharer.php?u=" + url,
    "share-email": "mailto:?subject=" + title + "&body=" + url
  };
  Object.keys(targets).forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.setAttribute("href", targets[id]);
  });
})();
