(function () {
  'use strict';

  function setupExpandable(el, fullText) {
    el.textContent = fullText;
    el.classList.add('prompt-clampable', 'prompt-clamped');

    requestAnimationFrame(function () {
      if (el.scrollHeight <= el.clientHeight + 1) {
        el.classList.remove('prompt-clamped');
        return;
      }

      var toggle = document.createElement('button');
      toggle.className = 'prompt-toggle';
      toggle.innerHTML = '<i class="fas fa-chevron-down"></i> more';
      el.after(toggle);

      toggle.addEventListener('click', function () {
        var clamped = el.classList.toggle('prompt-clamped');
        toggle.innerHTML = clamped
          ? '<i class="fas fa-chevron-down"></i> more'
          : '<i class="fas fa-chevron-up"></i> less';
      });
    });
  }

  var stages = document.querySelectorAll('.pipeline-stage');
  var panels = document.querySelectorAll('.stage-panel');
  var emptyState = document.getElementById('stage-panels-empty');

  function activateStage(stageName) {
    stages.forEach(function (s) {
      var isActive = s.dataset.stage === stageName;
      s.classList.toggle('active', isActive);
      s.setAttribute('aria-expanded', isActive ? 'true' : 'false');
    });
    panels.forEach(function (p) {
      var panelStage = p.id.replace('panel-', '');
      var isActive = panelStage === stageName;
      p.classList.toggle('active', isActive);
      p.hidden = !isActive;
    });
    if (emptyState) emptyState.style.display = stageName ? 'none' : 'flex';
  }

  function handleStageActivation(el) {
    var stage = el.dataset.stage;
    if (el.classList.contains('active')) {
      el.classList.remove('active');
      var panel = document.getElementById('panel-' + stage);
      panel.classList.remove('active');
      panel.hidden = true;
      el.setAttribute('aria-expanded', 'false');
      if (emptyState) emptyState.style.display = 'flex';
    } else {
      activateStage(stage);
    }
  }

  stages.forEach(function (el) {
    el.addEventListener('click', function () {
      handleStageActivation(el);
    });
    el.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleStageActivation(el);
      }
    });
  });

  activateStage('');

  function syncVideoGroup(videos) {
    var total = videos.length;
    if (total < 2) return;
    var readyCount = 0;
    var endedCount = 0;
    var activeCount = total;

    function playAll() {
      for (var i = 0; i < total; i++) {
        videos[i].currentTime = 0;
        var p = videos[i].play();
        if (p && p.catch) p.catch(function () {});
      }
    }

    function checkReady() {
      readyCount++;
      if (readyCount >= activeCount) playAll();
    }

    function checkEnded() {
      endedCount++;
      if (endedCount >= activeCount) {
        endedCount = 0;
        playAll();
      }
    }

    function handleError() {
      activeCount--;
      if (activeCount > 0 && readyCount >= activeCount) playAll();
    }

    for (var i = 0; i < total; i++) {
      videos[i].addEventListener('ended', checkEnded);
      videos[i].addEventListener('error', handleError, { once: true });
      if (videos[i].readyState >= 3) {
        checkReady();
      } else {
        videos[i].addEventListener('canplay', checkReady, { once: true });
      }
    }
  }

  var grid = document.getElementById('comparison-grid');
  var gallery = document.getElementById('examples-gallery');
  var refinementContainer = document.getElementById('refinement-workflows');

  fetch('assets/results/manifest.json')
    .then(function (r) { return r.json(); })
    .then(function (manifest) {

      // --- Comparison with Puppeteer (generic models that have a puppeteer render) ---
      if (grid) {
        manifest.generic.forEach(function (entry) {
          if (!entry.has_puppeteer) return;

          var puppeteerSrc = 'assets/results/puppeteer/' + entry.video;
          var oursSrc = 'assets/results/ours/generic/' + entry.video;
          var metaFile = 'assets/results/metadata/generic/' + entry.name + '.json';

          var row = document.createElement('div');
          row.className = 'comparison-row';

          var pair = document.createElement('div');
          pair.className = 'comparison-pair';

          pair.innerHTML =
            '<div class="video-cell">' +
              '<p class="video-label">Puppeteer</p>' +
              '<video muted playsinline><source src="' + puppeteerSrc + '" type="video/mp4"></video>' +
            '</div>' +
            '<div class="video-cell">' +
              '<p class="video-label">Ours</p>' +
              '<video muted playsinline><source src="' + oursSrc + '" type="video/mp4"></video>' +
            '</div>';

          var prompt = document.createElement('p');
          prompt.className = 'prompt-text';
          prompt.textContent = 'Loading prompt\u2026';

          fetch(metaFile)
            .then(function (r) { return r.json(); })
            .then(function (meta) {
              prompt.textContent = '\u201C' + meta.prompt + '\u201D';
            })
            .catch(function () {
              prompt.textContent = entry.name.replace(/_/g, ' ');
            });

          row.appendChild(pair);
          row.appendChild(prompt);
          grid.appendChild(row);
          syncVideoGroup(Array.prototype.slice.call(pair.querySelectorAll('video')));
        });
      }

      // --- Iterative Refinement workflows (carousel) ---
      if (refinementContainer && manifest.refinements) {
        manifest.refinements.forEach(function (workflow) {
          var metaFile = 'assets/results/metadata/refinements/' + workflow.name + '.json';
          var numSteps = workflow.steps.length;
          var basePath = 'assets/results/ours/refinements/' + workflow.name + '/';

          function stepLabel(i) { return i === 0 ? 'Initial' : 'Refinement ' + i; }

          function buildPair(leftIdx, rightIdx) {
            var pair = document.createElement('div');
            pair.className = 'refinement-pair';

            var left = document.createElement('div');
            left.className = 'refinement-step';
            var ll = document.createElement('p');
            ll.className = 'refinement-label';
            ll.textContent = stepLabel(leftIdx);
            var lv = document.createElement('video');
            lv.className = 'refinement-video';
            lv.muted = true; lv.playsInline = true;
            lv.innerHTML = '<source src="' + basePath + workflow.steps[leftIdx] + '" type="video/mp4">';
            left.appendChild(ll);
            left.appendChild(lv);

            var arrow = document.createElement('div');
            arrow.className = 'refinement-arrow-mid';
            arrow.innerHTML = '<i class="fas fa-arrow-right"></i>';

            var right = document.createElement('div');
            right.className = 'refinement-step';
            var rl = document.createElement('p');
            rl.className = 'refinement-label';
            rl.textContent = stepLabel(rightIdx);
            var rv = document.createElement('video');
            rv.className = 'refinement-video';
            rv.muted = true; rv.playsInline = true;
            rv.innerHTML = '<source src="' + basePath + workflow.steps[rightIdx] + '" type="video/mp4">';
            right.appendChild(rl);
            right.appendChild(rv);

            var rp = document.createElement('p');
            rp.className = 'refinement-prompt';
            rp.textContent = 'Loading\u2026';
            rp.dataset.stepIdx = rightIdx;

            pair.appendChild(left);
            pair.appendChild(arrow);
            pair.appendChild(right);
            pair.appendChild(rp);
            return pair;
          }

          // Build pages: each page shows a pair (step i, step i+1)
          var pages = [];
          for (var p = 0; p < numSteps - 1; p++) {
            pages.push({ left: p, right: p + 1 });
          }

          var carousel = document.createElement('div');
          carousel.className = 'refinement-carousel';

          var viewport = document.createElement('div');
          viewport.className = 'refinement-viewport';

          var track = document.createElement('div');
          track.className = 'refinement-track';

          pages.forEach(function (page) {
            var slide = document.createElement('div');
            slide.className = 'refinement-slide';
            slide.appendChild(buildPair(page.left, page.right));
            track.appendChild(slide);
          });

          viewport.appendChild(track);
          carousel.appendChild(viewport);

          // Side navigation (only if more than 1 page)
          var currentPage = 0;
          if (pages.length > 1) {
            var prevBtn = document.createElement('button');
            prevBtn.className = 'refinement-nav prev';
            prevBtn.innerHTML = '<i class="fas fa-chevron-left"></i>';
            prevBtn.disabled = true;

            var nextBtn = document.createElement('button');
            nextBtn.className = 'refinement-nav next';
            nextBtn.innerHTML = '<i class="fas fa-chevron-right"></i>';

            function goToPage(idx) {
              currentPage = idx;
              track.style.transform = 'translateX(-' + (idx * 100) + '%)';
              prevBtn.disabled = idx === 0;
              nextBtn.disabled = idx === pages.length - 1;

              // Play/pause videos based on visible page
              var slides = track.querySelectorAll('.refinement-slide');
              slides.forEach(function (s, si) {
                var vids = s.querySelectorAll('video');
                vids.forEach(function (v) {
                  if (si === idx) { v.currentTime = 0; v.play(); }
                  else { v.pause(); }
                });
              });
            }

            prevBtn.addEventListener('click', function () {
              if (currentPage > 0) goToPage(currentPage - 1);
            });
            nextBtn.addEventListener('click', function () {
              if (currentPage < pages.length - 1) goToPage(currentPage + 1);
            });

            carousel.appendChild(prevBtn);
            carousel.appendChild(nextBtn);
          }

          refinementContainer.appendChild(carousel);

          // Sync videos on the first visible page
          var firstSlideVideos = track.querySelector('.refinement-slide').querySelectorAll('video');
          syncVideoGroup(Array.prototype.slice.call(firstSlideVideos));

          // Load prompts
          fetch(metaFile)
            .then(function (r) { return r.json(); })
            .then(function (meta) {
              var prompts = carousel.querySelectorAll('.refinement-prompt');
              prompts.forEach(function (el) {
                var i = parseInt(el.dataset.stepIdx, 10);
                if (meta.workflow_steps && meta.workflow_steps[i]) {
                  var text = meta.workflow_steps[i].step_prompt;
                  setupExpandable(el, '\u201C' + text + '\u201D');
                }
              });
            })
            .catch(function () {
              var prompts = carousel.querySelectorAll('.refinement-prompt');
              prompts.forEach(function (el) {
                el.textContent = workflow.name.replace(/_/g, ' ');
              });
            });
        });
      }

      // --- Additional Examples ---
      if (gallery) {
        manifest.additional_examples.forEach(function (entry) {
          var videoSrc = 'assets/results/ours/eney/' + entry.video;
          var metaFile = 'assets/results/metadata/eney/' + entry.name + '.json';

          var card = document.createElement('div');
          card.className = 'example-card';

          var video = document.createElement('video');
          video.className = 'gallery-video';
          video.autoplay = true;
          video.loop = true;
          video.muted = true;
          video.playsInline = true;
          video.innerHTML = '<source src="' + videoSrc + '" type="video/mp4">';

          var prompt = document.createElement('p');
          prompt.className = 'prompt-text';
          prompt.textContent = 'Loading\u2026';

          var tagsContainer = document.createElement('div');
          tagsContainer.className = 'example-tags';

          fetch(metaFile)
            .then(function (r) { return r.json(); })
            .then(function (meta) {
              setupExpandable(prompt, '\u201C' + meta.prompt + '\u201D');

              if (meta.refined || meta.incremental_generation) {
                var tag = document.createElement('span');
                tag.className = 'example-tag tag-refined';
                tag.textContent = 'Refined';
                tagsContainer.appendChild(tag);
              }
              if (meta.first_try) {
                var tag = document.createElement('span');
                tag.className = 'example-tag tag-first-try';
                tag.textContent = 'First try';
                tagsContainer.appendChild(tag);
              }
              if (meta.no_examples) {
                var tag = document.createElement('span');
                tag.className = 'example-tag tag-no-examples';
                tag.textContent = 'No examples';
                tagsContainer.appendChild(tag);
              }
            })
            .catch(function () {
              prompt.textContent = entry.name.replace(/_/g, ' ');
            });

          card.appendChild(video);
          card.appendChild(tagsContainer);
          card.appendChild(prompt);
          gallery.appendChild(card);
        });
      }
    })
    .catch(function (err) {
      if (grid) grid.innerHTML = '<p class="has-text-grey">Could not load comparison results.</p>';
      if (gallery) gallery.innerHTML = '<p class="has-text-grey">Could not load examples.</p>';
      console.error('Failed to load manifest.json:', err);
    });

})();
