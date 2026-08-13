/* ==========================================================================
   ARC REACTOR — scroll-driven, depth-based 3D monochrome experience
   Three.js builds a detailed reactor · anime.js drives scroll choreography
   Scrolling through "EXPLODED ASSEMBLY" disassembles the reactor piece by piece
   ========================================================================== */
(function () {
  "use strict";

  const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (k) => k * k * (3 - 2 * k);

  /* ======================================================================
     1. THREE.JS — THE ARC REACTOR (detailed, line-art "sketch" style)
     ====================================================================== */

  const canvas = document.getElementById("reactor");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    55, window.innerWidth / window.innerHeight, 0.1, 200
  );
  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);

  // --- lighting ----------------------------------------------------------
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(6, 4, 8);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5);
  fill.position.set(-5, -3, 4);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffffff, 0.9);
  rim.position.set(-4, 5, -6);
  scene.add(rim);
  const coreLight = new THREE.PointLight(0xffffff, 3.2, 8, 1.6);
  coreLight.position.set(0, 0, 0.3);
  scene.add(coreLight);

  // --- materials ---------------------------------------------------------
  const matDark = new THREE.MeshStandardMaterial({ color: 0x161616, metalness: 0.7, roughness: 0.42 });
  const matMid  = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, metalness: 0.6, roughness: 0.38 });
  const matLite = new THREE.MeshStandardMaterial({ color: 0x3a3a3a, metalness: 0.55, roughness: 0.34 });
  const matCore = new THREE.MeshBasicMaterial({ color: 0xffffff });

  function edgeLine(color, opacity) {
    return new THREE.LineBasicMaterial({ color, transparent: true, opacity: opacity || 0.85 });
  }
  // add crisp white edge outlines for the "sketch" look
  function sketch(mesh, opacity) {
    const e = new THREE.EdgesGeometry(mesh.geometry, 22);
    const l = new THREE.LineSegments(e, edgeLine(0xffffff, opacity));
    mesh.add(l);
    return mesh;
  }

  function glowTexture(size) {
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const ctx = c.getContext("2d");
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0.0, "rgba(255,255,255,1)");
    g.addColorStop(0.16, "rgba(255,255,255,0.75)");
    g.addColorStop(0.4, "rgba(255,255,255,0.18)");
    g.addColorStop(1.0, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    return new THREE.CanvasTexture(c);
  }

  // --- coil fin geometry (tapered wedge pointing inward) -----------------
  function coilGeo(outerR, innerR, halfW, depth) {
    const s = new THREE.Shape();
    s.moveTo(-halfW, outerR);
    s.lineTo(halfW, outerR);
    s.lineTo(0, innerR);
    s.closePath();
    return new THREE.ExtrudeGeometry(s, {
      depth, bevelEnabled: true,
      bevelThickness: 0.035, bevelSize: 0.02, bevelSegments: 1,
    });
  }

  function radialLineSegments(count, innerR, outerR, opacity) {
    const pts = [];
    for (let i = 0; i < count; i++) {
      const a = (i / count) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * innerR, Math.sin(a) * innerR, 0));
      pts.push(new THREE.Vector3(Math.cos(a) * outerR, Math.sin(a) * outerR, 0));
    }
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    return new THREE.LineSegments(g, edgeLine(0xffffff, opacity));
  }

  const reactor = new THREE.Group();
  scene.add(reactor);

  // ambient faint frame rings (not exploded)
  const amb1 = sketch(new THREE.Mesh(new THREE.TorusGeometry(3.55, 0.006, 8, 120), new THREE.MeshBasicMaterial({ color: 0x1a1a1a })), 0.5);
  const amb2 = sketch(new THREE.Mesh(new THREE.TorusGeometry(4.0, 0.004, 8, 120), new THREE.MeshBasicMaterial({ color: 0x141414 })), 0.4);
  reactor.add(amb1, amb2);

  // particle fields
  function makePoints(count, radius, size) {
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = radius * (0.5 + Math.random() * 0.5);
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3 + 0] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = (Math.random() - 0.5) * 1.6;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0xffffff, size, map: glowTexture(64),
      transparent: true, opacity: 0.55, depthWrite: false,
      blending: THREE.AdditiveBlending, sizeAttenuation: true,
    });
    return new THREE.Points(geo, mat);
  }
  const dust = makePoints(360, 9, 0.045);
  const sparks = makePoints(220, 4.2, 0.028);
  scene.add(dust, sparks);

  /* ======================================================================
     COMPONENTS — each is a group that will explode along Z
     ====================================================================== */

  const parts = [];
  function register(group, anchor, zA, zB, start, end, meta) {
    group.position.z = zA;
    reactor.add(group);
    parts.push({ group, anchor, zA, zB, start, end, ...meta });
  }

  // ---- 6 · PALLADIUM CORE -------------------------------------------------
  {
    const g = new THREE.Group();
    const core = sketch(new THREE.Mesh(new THREE.IcosahedronGeometry(0.58, 1), matCore), 0.95);
    core.material.flatShading = true;
    g.add(core);
    // crystal facets ring
    const ring = sketch(new THREE.Mesh(new THREE.TorusGeometry(0.72, 0.03, 12, 72), matMid), 0.9);
    g.add(ring);
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTexture(256), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
    }));
    glow.scale.setScalar(3.2);
    g.add(glow);
    const anchor = new THREE.Object3D();
    g.add(anchor);
    register(g, anchor, 0.34, 4.1, 0.66, 0.88, {
      num: "06", name: "PALLADIUM CORE",
      desc: "The heart — a palladium catalyst sustaining a continuous fusion reaction.",
    });
  }

  // ---- 5 · CONFINEMENT RING ----------------------------------------------
  {
    const g = new THREE.Group();
    g.add(sketch(new THREE.Mesh(new THREE.TorusGeometry(1.06, 0.055, 16, 90), matMid), 0.9));
    g.add(sketch(new THREE.Mesh(new THREE.TorusGeometry(0.9, 0.04, 14, 90), matLite), 0.85));
    // inner segmented ring
    g.add(sketch(new THREE.Mesh(new THREE.TorusGeometry(1.28, 0.03, 10, 40), matDark), 0.6));
    // 6 support struts
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2;
      const strut = sketch(new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.16, 0.08), matLite), 0.9);
      strut.position.set(Math.cos(a) * 1.17, Math.sin(a) * 1.17, 0);
      strut.rotation.z = a + Math.PI / 2;
      g.add(strut);
    }
    const anchor = new THREE.Object3D(); anchor.position.set(1.06, 0, 0); g.add(anchor);
    register(g, anchor, 0.28, 2.9, 0.52, 0.72, {
      num: "05", name: "CONFINEMENT RING",
      desc: "Magnetic field lattice holding the plasma in a stable toroid.",
    });
  }

  // ---- 4 · INNER COIL RING ------------------------------------------------
  {
    const g = new THREE.Group();
    const geo = coilGeo(1.82, 1.02, 0.14, 0.16);
    for (let i = 0; i < 10; i++) {
      const c = sketch(new THREE.Mesh(geo, matLite), 0.85);
      c.rotation.z = (i / 10) * Math.PI * 2;
      g.add(c);
    }
    g.add(sketch(new THREE.Mesh(new THREE.TorusGeometry(1.56, 0.03, 12, 80), matDark), 0.6));
    const anchor = new THREE.Object3D(); anchor.position.set(1.82, 0, 0); g.add(anchor);
    register(g, anchor, 0.2, 1.6, 0.38, 0.58, {
      num: "04", name: "INNER COIL RING",
      desc: "Secondary coils focusing the energy inward toward the core.",
    });
  }

  // ---- 3 · CONCENTRIC SHIELD RINGS ----------------------------------------
  {
    const g = new THREE.Group();
    g.add(sketch(new THREE.Mesh(new THREE.TorusGeometry(2.4, 0.06, 16, 110), matMid), 0.85));
    g.add(sketch(new THREE.Mesh(new THREE.TorusGeometry(2.1, 0.045, 14, 110), matLite), 0.8));
    // 12 capacitor segments between the rings
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const cap = sketch(new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.3, 12), matDark), 0.75);
      cap.position.set(Math.cos(a) * 2.25, Math.sin(a) * 2.25, 0);
      cap.rotation.z = a + Math.PI / 2;
      g.add(cap);
    }
    const anchor = new THREE.Object3D(); anchor.position.set(2.4, 0, 0); g.add(anchor);
    register(g, anchor, 0.12, 0.3, 0.24, 0.44, {
      num: "03", name: "CONCENTRIC RINGS",
      desc: "Shielding rings stepping the field down in voltage.",
    });
  }

  // ---- 2 · OUTER COIL RING ------------------------------------------------
  {
    const g = new THREE.Group();
    const geo = coilGeo(2.62, 1.5, 0.22, 0.2);
    for (let i = 0; i < 10; i++) {
      const c = sketch(new THREE.Mesh(geo, matLite), 0.9);
      c.rotation.z = (i / 10) * Math.PI * 2;
      g.add(c);
    }
    g.add(sketch(new THREE.Mesh(new THREE.TorusGeometry(2.62, 0.045, 14, 110), matDark), 0.6));
    const anchor = new THREE.Object3D(); anchor.position.set(2.62, 0, 0); g.add(anchor);
    register(g, anchor, 0.05, -0.6, 0.1, 0.3, {
      num: "02", name: "OUTER COIL RING",
      desc: "Ten radial coils projecting the field outward.",
    });
  }

  // ---- 1 · TITANIUM CASING (backplate) -------------------------------------
  {
    const g = new THREE.Group();
    g.add(sketch(new THREE.Mesh(new THREE.CylinderGeometry(3.0, 3.0, 0.12, 72), matDark), 0.85));
    g.add(sketch(new THREE.Mesh(new THREE.TorusGeometry(3.0, 0.07, 14, 110), matMid), 0.9));
    // 12 bolts
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const bolt = sketch(new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.16, 12), matLite), 0.8);
      bolt.position.set(Math.cos(a) * 2.72, Math.sin(a) * 2.72, 0.08);
      g.add(bolt);
    }
    // radial spoke lines (technical drawing detail)
    g.add(radialLineSegments(16, 0.5, 2.95, 0.5));
    g.add(radialLineSegments(8, 1.4, 2.9, 0.35));
    const anchor = new THREE.Object3D(); anchor.position.set(3.0, 0, 0); g.add(anchor);
    register(g, anchor, -0.6, -1.8, 0.0, 0.16, {
      num: "01", name: "TITANIUM CASING",
      desc: "Outer housing — triple-sealed against the void.",
    });
  }

  /* ======================================================================
     2. LABELS — projected 2D callouts tracking each 3D part
     ====================================================================== */

  const labelLayer = document.createElement("div");
  labelLayer.id = "explode-labels";
  document.body.appendChild(labelLayer);

  parts.forEach((p) => {
    const el = document.createElement("div");
    el.className = "part-label";
    el.innerHTML =
      '<span class="pl-dot"></span>' +
      '<span class="pl-line"></span>' +
      '<span class="pl-text"><em>' + p.num + '</em><b>' + p.name + '</b></span>';
    el.style.opacity = "0";
    labelLayer.appendChild(el);
    p.labelEl = el;
  });

  const captionNum = document.getElementById("explode-caption-num");
  const captionName = document.getElementById("explode-caption-name");
  const captionDesc = document.getElementById("explode-caption-desc");
  const assemblyFill = document.getElementById("explode-assembly-fill");

  const tmpV = new THREE.Vector3();
  function project(anchor) {
    anchor.getWorldPosition(tmpV);
    tmpV.project(camera);
    return {
      x: (tmpV.x * 0.5 + 0.5) * window.innerWidth,
      y: (-tmpV.y * 0.5 + 0.5) * window.innerHeight,
      behind: tmpV.z > 1,
    };
  }

  /* ======================================================================
     3. SCROLL STATE + REACTOR POSES
     ====================================================================== */

  let scrollProgress = 0;     // 0..1 across whole page
  let targetProgress = 0;
  let explodeTarget = 0;      // 0..1 within anatomy section
  let explode = 0;
  let mouseX = 0, mouseY = 0;

  const poses = [
    { t: 0.00, camZ: 10.0, rotX: 0.12, rotY: 0.6,  px: 0,    py: 0,    scale: 1.0,  glow: 1.0 },
    { t: 0.16, camZ: 12.0, rotX: 0.3,  rotY: 1.9,  px: 0,    py: 0.4,  scale: 0.9,  glow: 0.8 },
    { t: 0.36, camZ: 13.5, rotX: 0.5,  rotY: 3.4,  px: 1.6,  py: 0,    scale: 0.7,  glow: 0.9 },
    { t: 0.56, camZ: 12.5, rotX: -0.35,rotY: 4.9,  px: -1.6, py: 0.2,  scale: 0.78, glow: 1.0 },
    { t: 0.76, camZ: 14.0, rotX: 0.35, rotY: 6.2,  px: 0,    py: -0.4, scale: 0.6,  glow: 0.7 },
    { t: 1.00, camZ: 9.5,  rotX: 0.0,  rotY: 8.4,  px: 0,    py: 0,    scale: 1.15, glow: 1.5 },
  ];

  function poseAt(p) {
    let a = poses[0], b = poses[poses.length - 1];
    for (let i = 0; i < poses.length - 1; i++) {
      if (p >= poses[i].t && p <= poses[i + 1].t) { a = poses[i]; b = poses[i + 1]; break; }
    }
    const span = b.t - a.t || 1;
    const k = smooth(clamp((p - a.t) / span, 0, 1));
    const out = {};
    for (const key in a) { if (key === "t") continue; out[key] = lerp(a[key], b[key], k); }
    return out;
  }

  let surge = 0;

  /* ======================================================================
     4. ANIME.JS — SCROLL CHOREOGRAPHY (text reveals + counters)
     ====================================================================== */

  const animProps = {
    reveal: { opacity: [0, 1], translateY: [14, 0] },
    rise:   { opacity: [0, 1], translateY: [40, 0] },
    head:   { opacity: [0, 1], translateX: [-40, 0] },
    tline:  { opacity: [0, 1], translateY: [30, 0] },
    cta:    { opacity: [0, 1], scale: [0.7, 1] },
    grow:   { scaleX: [0, 1] },
    word:   { opacity: [0, 1], translateY: [60, 0] },
  };

  function splitWords(root) {
    function walk(node) {
      const frag = document.createDocumentFragment();
      node.childNodes.forEach((child) => {
        if (child.nodeType === 3) {
          child.textContent.split(/(\s+)/).forEach((tok) => {
            if (tok === "") return;
            if (/^\s+$/.test(tok)) { frag.appendChild(document.createTextNode(" ")); return; }
            const span = document.createElement("span");
            span.className = "word"; span.textContent = tok;
            frag.appendChild(span);
          });
        } else if (child.nodeType === 1) {
          const el = document.createElement(child.tagName);
          for (const a of child.attributes) el.setAttribute(a.name, a.value);
          el.appendChild(walk(child));
          frag.appendChild(el);
        }
      });
      return frag;
    }
    root.replaceWith(walk(root));
  }
  document.querySelectorAll(".statement, .final-statement").forEach(splitWords);

  const scenes = [];
  const DUR = 820, STAGGER = 130;
  function buildSectionScene(section) {
    const items = Array.from(
      section.querySelectorAll("[data-anim], .statement .word, .final-statement .word")
    );
    if (!items.length) return;
    const tl = anime.timeline({ autoplay: false });
    items.forEach((el, i) => {
      const type = el.dataset.anim || "word";
      tl.add({ targets: el, ...animProps[type], duration: DUR, easing: "easeOutExpo" }, i * STAGGER);
    });
    scenes.push({ section, tl });
  }
  ["vision", "specs", "anatomy", "timeline", "future"].forEach((id) => {
    const sec = document.getElementById(id);
    if (sec) buildSectionScene(sec);
  });

  // hero intro
  const heroIntro = anime.timeline({ easing: "easeOutExpo" });
  heroIntro
    .add({ targets: ".nav [data-anim='reveal']", opacity: [0, 1], translateY: [14, 0], duration: 700, delay: anime.stagger(70) }, 0)
    .add({ targets: ".hero .hero-title .word", opacity: [0, 1], translateY: ["60%", 0], duration: 1300 }, 250)
    .add({ targets: '.hero [data-anim="grow"]', scaleX: [0, 1], duration: 1200, easing: "easeInOutQuad" }, 700)
    .add({ targets: '.hero [data-anim="rise"]', opacity: [0, 1], translateY: [40, 0], duration: 1000, delay: anime.stagger(160) }, 900)
    .add({ targets: '.hero [data-anim="cue"]', opacity: [0, 1], translateY: [20, 0], duration: 900 }, 1500);

  // counters
  function animateCounter(el) {
    const target = el.querySelector("[data-target]");
    const value = parseFloat(el.dataset.value);
    const decimals = parseInt(el.dataset.decimals || "0", 10);
    const suffix = el.dataset.suffix || "";
    const obj = { n: 0 };
    anime({
      targets: obj, n: value, duration: 2200, easing: "easeOutExpo",
      update: () => {
        target.innerHTML = obj.n.toFixed(decimals) + '<span class="suffix">' + suffix + "</span>";
      },
    });
  }
  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { animateCounter(e.target); counterObserver.unobserve(e.target); }
    });
  }, { threshold: 0.6 });
  document.querySelectorAll("[data-counter]").forEach((el) => counterObserver.observe(el));

  function sectionProgress(el) {
    const r = el.getBoundingClientRect();
    const vh = window.innerHeight;
    return clamp((vh - r.top) / (r.height + vh), 0, 1);
  }

  // full 0..1 across the visible scroll of a pinned section
  function pinProgress(el) {
    const r = el.getBoundingClientRect();
    const range = el.offsetHeight - window.innerHeight;
    if (range <= 0) return 0;
    return clamp(-r.top / range, 0, 1);
  }

  let ticking = false;
  function updateScroll() {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    targetProgress = max > 0 ? clamp(window.scrollY / max, 0, 1) : 0;

    scenes.forEach((s) => {
      const p = sectionProgress(s.section);
      s.tl.seek(p * s.tl.duration);
    });

    const anat = document.getElementById("anatomy");
    if (anat) {
      const r = anat.getBoundingClientRect();
      if (r.bottom < 0) explodeTarget = 0;      // past the section → reassemble
      else explodeTarget = pinProgress(anat);
    }

    const pct = Math.round(targetProgress * 100);
    document.querySelector(".hud-progress-bar").style.height = (targetProgress * 100) + "%";
    document.querySelector(".hud-progress-label").textContent =
      "SCROLL // " + String(pct).padStart(3, "0");

    const ids = ["vision", "specs", "anatomy", "timeline", "future"];
    let active = -1;
    ids.forEach((id, i) => {
      const el = document.getElementById(id);
      const r = el.getBoundingClientRect();
      if (r.top < window.innerHeight * 0.6 && r.bottom > window.innerHeight * 0.4) active = i;
    });
    document.querySelectorAll(".hud-nav a").forEach((dot, i) => dot.classList.toggle("active", i === active));

    ticking = false;
  }
  function requestScroll() { if (!ticking) { ticking = true; requestAnimationFrame(updateScroll); } }
  window.addEventListener("scroll", requestScroll, { passive: true });

  /* ======================================================================
     5. RENDER LOOP
     ====================================================================== */

  const clock = new THREE.Clock();

  function render() {
    const dt = clock.getDelta();
    const t = clock.elapsedTime;

    scrollProgress = lerp(scrollProgress, targetProgress, 0.12);
    if (Math.abs(scrollProgress - targetProgress) < 0.0005) scrollProgress = targetProgress;
    explode = lerp(explode, explodeTarget, 0.1);
    if (Math.abs(explode - explodeTarget) < 0.001) explode = explodeTarget;

    const pose = poseAt(scrollProgress);
    const spin = lerp(0.14, 0.04, explode);

    // camera (dolly back as it explodes)
    const camZ = lerp(pose.camZ, 14.5, explode);
    const par = lerp(1, 0.35, explode);
    camera.position.x = pose.px * par + mouseX * 0.5 * par;
    camera.position.y = pose.py * par - mouseY * 0.4 * par;
    camera.position.z = camZ;
    camera.lookAt(0, -explode * 0.15, explode * 0.9);

    // reactor transform
    reactor.rotation.x = pose.rotX + Math.sin(t * 0.4) * 0.04;
    reactor.rotation.y = pose.rotY + spin * t;
    reactor.rotation.z = Math.cos(t * 0.33) * 0.03;
    const breathe = 1 + Math.sin(t * 1.6) * 0.02;
    reactor.scale.setScalar(pose.scale * breathe);

    // explode each part
    parts.forEach((p) => {
      const w = clamp((explode - p.start) / (p.end - p.start), 0, 1);
      p.group.position.z = lerp(p.zA, p.zB, smooth(w));
    });

    // labels + caption
    let activeIdx = -1, bestStart = -1;
    parts.forEach((p, i) => {
      if (explode >= p.start && p.start > bestStart) { bestStart = p.start; activeIdx = i; }
      const vis = clamp((explode - p.start) / 0.05, 0, 1);
      p.labelEl.style.opacity = vis;
      if (vis > 0.001) {
        const s = project(p.anchor);
        if (!s.behind) {
          p.labelEl.style.transform = "translate(" + s.x.toFixed(1) + "px," + s.y.toFixed(1) + "px)";
        }
      }
    });

    if (explode < 0.02) {
      if (captionName) captionName.textContent = "FULLY ASSEMBLED";
      if (captionDesc) captionDesc.textContent = "All components locked — reactor stable at 100% output.";
      if (captionNum) captionNum.textContent = "00";
    } else if (activeIdx >= 0) {
      const p = parts[activeIdx];
      if (captionName) captionName.textContent = p.name;
      if (captionDesc) captionDesc.textContent = p.desc;
      if (captionNum) captionNum.textContent = p.num;
    }
    if (assemblyFill) assemblyFill.style.width = (explode * 100).toFixed(1) + "%";

    // core glow + light
    const glowScale = pose.glow * (1 + Math.sin(t * 2.2) * 0.06 + surge * 1.4);
    const coreGroup = parts[0].group;
    coreGroup.children.forEach((c) => {
      if (c.isSprite) c.scale.setScalar(3.2 * glowScale);
    });
    coreLight.intensity = 3.2 + surge * 8;

    dust.rotation.y += dt * 0.02;
    sparks.rotation.z += dt * 0.04;
    sparks.rotation.y += dt * 0.02;

    if (surge > 0) surge = Math.max(0, surge - dt * 0.5);

    renderer.render(scene, camera);
    requestAnimationFrame(render);
  }
  render();

  /* ======================================================================
     6. INPUT + RESIZE + CTA
     ====================================================================== */

  window.addEventListener("mousemove", (e) => {
    mouseX = (e.clientX / window.innerWidth) * 2 - 1;
    mouseY = (e.clientY / window.innerHeight) * 2 - 1;
  });
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  const flash = document.createElement("div");
  flash.style.cssText = "position:fixed;inset:0;background:#fff;opacity:0;z-index:60;pointer-events:none;";
  document.body.appendChild(flash);

  const cta = document.querySelector(".cta");
  if (cta) {
    cta.addEventListener("click", () => {
      surge = 1;
      anime({ targets: flash, opacity: [0, 0.22, 0], duration: 900, easing: "easeOutQuad" });
      anime({ targets: ".cta-ring", scale: [1, 1.25, 1], duration: 900, easing: "easeOutElastic(1, .4)" });
      const mono = document.querySelector(".cta-mono");
      if (mono) {
        mono.textContent = "SEQUENCE INITIATED // CORE STABLE";
        setTimeout(() => { mono.textContent = "AWAITING OPERATOR INPUT . . ."; }, 2600);
      }
    });
  }

  updateScroll();
})();
