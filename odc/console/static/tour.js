/* ODC interactive guided tour — real UI, step by step.
 *
 * Unlike a static popup, this tour anchors every step to a live element:
 * it scrolls the target into view, outlines it, and lets you keep using
 * the page (the dim layer never intercepts clicks). Steps that live on
 * another page navigate there and resume automatically.
 */
(function () {
  'use strict';

  var KEY = 'odc-tour-step';

  var STEPS = [
    {
      page: '/live', sel: '#runform button[type="submit"]',
      title: '1 · Start a session',
      text: 'Pick a clip (or upload your own), choose a layout or leave auto, then Start. The server analyses the video for seat mapping first — the status line tells you which layout won.',
    },
    {
      page: '/live', sel: '#stage-box',
      title: '2 · Live stream is the truth',
      text: 'Real frames with seat zones, people, belt straps and masks overlaid. Toggle overlays without affecting detection — analytics always run underneath.',
    },
    {
      page: '/live', sel: '#seat-chips',
      title: '3 · Seat truth chips',
      text: 'One chip per seat: presence dot, belt state (belt! = issue) and M/F appearance. Click any chip to open its evidence in the inspector.',
    },
    {
      page: '/live', sel: '#alerts-card',
      title: '4 · Alerts live here',
      text: 'Active alerts plus recent history, each anchored to its seat and video frame. Click a seat name to jump to it.',
    },
    {
      page: '/seats', sel: '.seatview-btn[data-view="poly"]',
      title: '5 · Zones over the frame',
      text: 'Schematic arranges seats as a cabin; Zones shows true calibrated polygons over the live frame. Both follow the active session.',
    },
    {
      page: '/seats', sel: '#mapedit-box',
      title: '6 · Trace and refine zones',
      text: 'Capture a frame, then trace seats (click points or drag a rectangle), drag zones to move them, drag corners to resize. Overlaps warn live in red.',
    },
    {
      page: '/seats', sel: '#me-save-name',
      title: '7 · Save applies instantly',
      text: 'Save stores the layout in PLOMID and applies it at once — a running session hot-swaps without restart, keeping temporal state for unchanged seats.',
    },
    {
      page: '/timeline', sel: '#tl-kind',
      title: '8 · Timeline analysis',
      text: 'Filter by seat, kind and severity. Occupancy, belt and alert markers share one evidence clock — click any event to inspect it and jump to its video frame.',
    },
    {
      page: '/timeline', sel: '#ev-frame',
      title: '9 · Event to frame',
      text: 'Select an event, then open its exact video frame here. That is the full loop: pixels → seats → events → back to pixels.',
    },
  ];

  function storeGet() {
    try {
      var v = localStorage.getItem(KEY);
      return v === null ? null : parseInt(v, 10);
    } catch (e) {
      return null;
    }
  }
  function storeSet(i) {
    try {
      if (i === null || i === undefined) localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, String(i));
    } catch (e) {}
  }

  function el() {
    return document.getElementById('tour-root');
  }
  function clear() {
    var root = el();
    if (root) root.innerHTML = '';
    var dim = document.getElementById('tour-dim');
    if (dim && dim.parentNode) dim.parentNode.removeChild(dim);
    document.querySelectorAll('.tour-target').forEach(function (n) {
      n.classList.remove('tour-target');
    });
  }

  function targetFor(step) {
    try {
      return document.querySelector(step.sel);
    } catch (e) {
      return null;
    }
  }

  function show(i) {
    clear();
    if (i < 0 || i >= STEPS.length) {
      storeSet(null);
      return;
    }
    var step = STEPS[i];
    if (window.location.pathname !== step.page) {
      storeSet(i);
      window.location.href = step.page;
      return;
    }
    storeSet(i);
    var node = targetFor(step);
    var root = el();
    if (!root) {
      storeSet(null);
      return;
    }
    var dim = document.createElement('div');
    dim.id = 'tour-dim';
    document.body.appendChild(dim);
    var rect = null;
    if (node) {
      try {
        node.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'smooth' });
      } catch (e) {}
      node.classList.add('tour-target');
      rect = node.getBoundingClientRect();
    }
    var box = document.createElement('div');
    box.className = 'tour-box';
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-label', step.title);
    box.innerHTML =
      '<div class="tour-progress">Guided tour · ' +
      (i + 1) +
      ' of ' +
      STEPS.length +
      '</div>' +
      '<h3>' +
      step.title +
      '</h3><p>' +
      step.text +
      '</p>' +
      '<div class="tour-nav"></div>';
    // Place the nav row inside the box; position the box itself.
    var wrap = document.createElement('div');
    wrap.className = 'tour-fixed';
    if (rect) {
      var ftop = rect.bottom + 12;
      var fmax = window.innerHeight - 230;
      if (ftop > fmax) ftop = Math.max(12, rect.top - 210);
      var fleft = Math.min(Math.max(12, rect.left), Math.max(12, window.innerWidth - 392));
      wrap.setAttribute('style', 'top:' + Math.round(ftop) + 'px;left:' + Math.round(fleft) + 'px');
    }
    box.querySelector('.tour-nav').outerHTML =
      '<div class="tour-nav">' +
      (i > 0 ? '<button type="button" class="ghost" data-tour="back">Back</button>' : '') +
      (i < STEPS.length - 1
        ? '<button type="button" data-tour="next">Next</button>'
        : '<button type="button" data-tour="done">Finish</button>') +
      ' <button type="button" class="ghost" data-tour="done">Skip tour</button></div>';
    wrap.appendChild(box);
    root.appendChild(wrap);
    wrap.addEventListener('click', function (e) {
      var b = e.target.closest && e.target.closest('[data-tour]');
      if (!b) return;
      var act = b.getAttribute('data-tour');
      if (act === 'next') show(i + 1);
      else if (act === 'back') show(i - 1);
      else {
        storeSet(null);
        clear();
      }
    });
  }

  function resume() {
    var i = storeGet();
    if (i !== null && i >= 0 && i < STEPS.length) {
      // Small delay so htmx/alpine have settled and targets exist.
      setTimeout(function () {
        show(i);
      }, 600);
    }
  }

  window.odcTour = {
    start: function () {
      show(0);
    },
    stop: function () {
      storeSet(null);
      clear();
    },
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', resume);
  } else {
    resume();
  }
})();
