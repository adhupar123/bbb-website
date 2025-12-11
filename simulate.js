/* simulate.js (Updated with RDKit for SMILES descriptor calculation + DEBUG MODE)
 - Loads RDKit to compute molecular descriptors from SMILES
 - Robust TSV loader & header matching
 - In-browser OLS trained on rows with experimental logBB
 - Fallback coefficients if OLS cannot be computed
 - Dropdown changes update sliders immediately
 - Two-compartment RK4 simulation for predicted & experimental LogBB
 - Plotly visualization
*/

(async function(){
  // -------- utils --------
  const log = (...args) => console.log('[simulate]', ...args);
  function toNum(x){ const v = (x === undefined || x === null || x === '') ? NaN : Number(String(x).trim()); return Number.isFinite(v) ? v : NaN; }
  async function fetchText(path){
    const r = await fetch(path);
    if(!r.ok) throw new Error(`Fetch failed ${path} ${r.status}`);
    return r.text();
  }

  // parse TSV - tolerate mixed separators, quotes, missing cols
  function parseTSV(txt){
    const lines = txt.replace(/\r/g,'').split('\n').map(l => l.trim()).filter(l => l.length>0);
    if(lines.length < 1) return {hdr:[], rows:[]};
    const hdr = lines[0].split(/\t/).map(h=>h.trim());
    const rows = lines.slice(1).map(line => {
      const cols = line.split(/\t/);
      const obj = {};
      hdr.forEach((h,i) => obj[h] = (i < cols.length) ? cols[i].trim() : '');
      return obj;
    });
    return {hdr, rows};
  }

  // simple matrix helpers for OLS
  function transpose(A){ return A[0].map((_,i) => A.map(r => r[i])); }
  function matMul(A,B){
    const m=A.length, n=A[0].length, p=B[0].length;
    const C = Array.from({length:m}, ()=> Array(p).fill(0));
    for(let i=0;i<m;i++){
      for(let k=0;k<n;k++){
        const aik = A[i][k];
        for(let j=0;j<p;j++) C[i][j] += aik * B[k][j];
      }
    }
    return C;
  }
  function invertMatrix(M){
    const n=M.length;
    const A=M.map(r=> r.slice());
    const I=Array.from({length:n}, (_,i)=> Array.from({length:n}, (__,j)=> i===j?1:0));
    for(let i=0;i<n;i++){
      if(Math.abs(A[i][i]) < 1e-12){
        let swap = i+1;
        while(swap<n && Math.abs(A[swap][i])<1e-12) swap++;
        if(swap === n) throw new Error('Singular matrix');
        [A[i], A[swap]] = [A[swap], A[i]];
        [I[i], I[swap]] = [I[swap], I[i]];
      }
      const piv = A[i][i];
      for(let j=0;j<n;j++){ A[i][j] /= piv; I[i][j] /= piv; }
      for(let r=0;r<n;r++){
        if(r===i) continue;
        const f = A[r][i];
        for(let c=0;c<n;c++){ A[r][c] -= f*A[i][c]; I[r][c] -= f*I[i][c]; }
      }
    }
    return I;
  }
  function ols(X, y){
    const Xt = transpose(X);
    const XtX = matMul(Xt, X);
    const XtXinv = invertMatrix(XtX);
    const Xty = matMul(Xt, y.map(v=>[v]));
    const betaMat = matMul(XtXinv, Xty);
    return betaMat.map(r => r[0]);
  }

  // RK4 two-compartment simulation (same equations as notebook)
  function simulateTwoComp(logBB, P_eff=0.15, Tmax=24, dt=0.05){
    const K = Math.pow(10, logBB);
    const steps = Math.max(2, Math.floor(Tmax/dt) + 1);
    const t = new Array(steps);
    const Cb = new Array(steps);
    const Cbr = new Array(steps);
    let cb = 1.0, cbr = 0.0;
    t[0] = 0; Cb[0] = cb; Cbr[0] = cbr;
    function rhs([cB, cBr]) {
      const dcb = -P_eff * (cB - (cBr / K));
      const dcbr =  P_eff * (cB - (cBr / K));
      return [dcb, dcbr];
    }
    for(let i=1;i<steps;i++){
      const dtlocal = dt;
      const s0 = [cb, cbr];
      const k1 = rhs(s0);
      const s1 = [ cb + 0.5*dtlocal*k1[0], cbr + 0.5*dtlocal*k1[1] ];
      const k2 = rhs(s1);
      const s2 = [ cb + 0.5*dtlocal*k2[0], cbr + 0.5*dtlocal*k2[1] ];
      const k3 = rhs(s2);
      const s3 = [ cb + dtlocal*k3[0], cbr + dtlocal*k3[1] ];
      const k4 = rhs(s3);
      cb  = cb + (dtlocal/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
      cbr = cbr + (dtlocal/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
      t[i] = i * dtlocal;
      Cb[i] = cb;
      Cbr[i] = cbr;
    }
    return {t, Cb, Cbr};
  }

  // -------- Wait for RDKit to load --------
  log('Waiting for RDKit to initialize...');
  await window.initRDKitModule();
  const RDKit = window.RDKitModule;
  log('RDKit loaded successfully');

  // Hide loading overlay
  const loadingEl = document.getElementById('loading');
  if(loadingEl) loadingEl.style.display = 'none';

  // Function to calculate descriptors from SMILES
  function calculateDescriptors(smiles) {
    if (!smiles || smiles === 'nan') return null;
    
    try {
      const mol = RDKit.get_mol(smiles);
      if (!mol || !mol.is_valid()) {
        log('Invalid molecule for SMILES:', smiles);
        return null;
      }

      const descriptors = JSON.parse(mol.get_descriptors());
      
      // Calculate aromatic rings
      const aromaticRings = mol.get_substruct_matches(RDKit.get_qmol('c1ccccc1')).length;
      
      const result = {
        MolLogP: descriptors.CrippenClogP || 0,
        MolWt: descriptors.amw || 0,
        TPSA: descriptors.TPSA || 0,
        FC: mol.get_total_formal_charge() || 0,
        Aromatic: aromaticRings,
        Heavy: mol.get_num_heavy_atoms() || 0
      };
      
      mol.delete();
      return result;
    } catch (e) {
      log('Error calculating descriptors for SMILES:', smiles, e);
      return null;
    }
  }

  // -------- UI references (IDs must match simulate.html) --------
  const ids = ['MolLogP','MolWt','TPSA','FC','Aromatic','Heavy'];
  const ui = {};
  ids.forEach(id => ui[id] = document.getElementById(id));
  ui.P_eff = document.getElementById('P_eff');
  ui.Tmax  = document.getElementById('Tmax');

  const outputs = {
    MolLogP: document.getElementById('MolLogP_val'),
    MolWt: document.getElementById('MolWt_val'),
    TPSA: document.getElementById('TPSA_val'),
    FC: document.getElementById('FC_val'),
    Aromatic: document.getElementById('Aromatic_val'),
    Heavy: document.getElementById('Heavy_val')
  };

  const drugSel = document.getElementById('drug-select');
  const runBtn = document.getElementById('runBtn');
  const resetBtn = document.getElementById('resetBtn');
  const modelLogbbEl = document.getElementById('model_logbb');
  const expLogbbEl = document.getElementById('exp_logbb');

  // DEBUG: Check if UI elements exist
  log('UI elements check:');
  ids.forEach(id => {
    log(`  ${id}: slider=${!!ui[id]}, output=${!!outputs[id]}`);
  });

  // ensure sliders show initial values
  ids.forEach(id => { if(ui[id]) outputs[id].textContent = ui[id].value; });

  // attach live input updates
  ids.forEach(id => {
    if(!ui[id]) return;
    ui[id].addEventListener('input', () => {
      outputs[id].textContent = ui[id].value;
      updateLogBBDisplay();
    });
  });

  // -------- load TSV and calculate descriptors --------
  let parsedRows = [];
  try {
    const raw = await fetchText('./B3DB_regression.tsv');
    const parsed = parseTSV(raw);
    const hdr = parsed.hdr;
    const rows = parsed.rows;

    log('TSV Headers:', hdr);

    // Find columns
    function findCol(candidates){
      for(const c of candidates){
        const exact = hdr.find(h => h === c);
        if(exact) return exact;
      }
      for(const c of candidates){
        const norm = c.toLowerCase().replace(/\W/g,'');
        const found = hdr.find(h => h.toLowerCase().replace(/\W/g,'') === norm);
        if(found) return found;
      }
      return null;
    }

    const col_name = findCol(['compound_name','compound','name']);
    const col_smiles = findCol(['SMILES','smiles','Smiles']);
    const col_logBB = findCol(['logBB','LogBB','logBB_exp','logBB_experimental','log_bb']);

    log('Column mappings:', {col_name, col_smiles, col_logBB});
    log('Processing', rows.length, 'compounds and calculating descriptors...');
    
    parsedRows = rows.map((r, idx) => {
      const smiles = r[col_smiles] || '';
      const descriptors = calculateDescriptors(smiles);
      
      if (idx < 3) {
        log(`Row ${idx}:`, {
          name: r[col_name],
          smiles: smiles.substring(0,50) + '...',
          descriptors: descriptors
        });
      }
      
      return {
        name: r[col_name] ?? r[Object.keys(r)[0]] ?? '',
        smiles: smiles,
        MolLogP: descriptors?.MolLogP ?? NaN,
        MolWt: descriptors?.MolWt ?? NaN,
        TPSA: descriptors?.TPSA ?? NaN,
        FC: descriptors?.FC ?? NaN,
        Aromatic: descriptors?.Aromatic ?? NaN,
        Heavy: descriptors?.Heavy ?? NaN,
        logBB_exp: col_logBB ? toNum(r[col_logBB]) : NaN
      };
    }).filter(r => r.name); // Keep only rows with names

    log('Processed', parsedRows.length, 'compounds with descriptors');
    log('Sample row:', parsedRows[0]);
  } catch (e) {
    log('Failed to load TSV:', e);
    document.getElementById('plot').innerHTML = `<div style="padding:12px;color:#900">Error: could not load B3DB_regression.tsv — check path and CORS in console.</div>`;
    return;
  }

  // populate dropdown
  const names = Array.from(new Set(parsedRows.map(r=> r.name))).filter(Boolean).sort();
  drugSel.innerHTML = '';
  const placeholderOpt = document.createElement('option');
  placeholderOpt.value = '';
  placeholderOpt.textContent = '-- Select drug --';
  drugSel.appendChild(placeholderOpt);
  names.forEach(n => { const o=document.createElement('option'); o.value=n; o.textContent=n; drugSel.appendChild(o); });

  log('Populated dropdown with', names.length, 'drugs');

  // Prepare regression training: only rows with numeric features AND experimental logBB
  const X_train = [];
  const y_train = [];
  parsedRows.forEach(r => {
    if([r.MolLogP, r.MolWt, r.TPSA, r.FC, r.Aromatic, r.Heavy].some(v => Number.isNaN(v))) return;
    if(Number.isNaN(r.logBB_exp)) return;
    X_train.push([1, r.MolLogP, r.MolWt, r.TPSA, r.FC, r.Aromatic, r.Heavy]);
    y_train.push(r.logBB_exp);
  });

  log('Training samples:', X_train.length);

  let beta = null;
  try {
    if(X_train.length >= 7){
      beta = ols(X_train, y_train);
      log('OLS coefficients:', beta);
    } else {
      log('Not enough training rows (need >=7), found', X_train.length);
    }
  } catch(err){
    log('OLS failed, will use fallback coefficients. Error:', err);
    beta = null;
  }

  // fallback coefficients (from About page equation)
  const fallbackBeta = [0.2407, -0.0115, -0.0013, -0.0133, 0.0172, -0.1323, 0.0512];

  function predictLogBBFromFeatures(features){
    const coeffs = (beta && Array.isArray(beta) && beta.every(c=> Number.isFinite(c))) ? beta : fallbackBeta;
    const vals = [1, features.MolLogP, features.MolWt, features.TPSA, features.FC, features.Aromatic, features.Heavy];
    let s = 0;
    for(let i=0;i<coeffs.length;i++) s += (coeffs[i] || 0) * (vals[i] || 0);
    return s;
  }

  function findRowByName(n){
    return parsedRows.find(r => r.name === n);
  }

  function applyRowToSliders(r){
    if(!r) {
      log('applyRowToSliders: row is null/undefined');
      return;
    }
    
    log('applyRowToSliders called with:', {
      name: r.name,
      MolLogP: r.MolLogP,
      MolWt: r.MolWt,
      TPSA: r.TPSA,
      FC: r.FC,
      Aromatic: r.Aromatic,
      Heavy: r.Heavy
    });
    
    // Store old values for comparison
    const oldValues = {
      MolLogP: ui.MolLogP.value,
      MolWt: ui.MolWt.value,
      TPSA: ui.TPSA.value,
      FC: ui.FC.value,
      Aromatic: ui.Aromatic.value,
      Heavy: ui.Heavy.value
    };
    
    if(Number.isFinite(r.MolLogP)) {
      ui.MolLogP.value = r.MolLogP;
      outputs.MolLogP.textContent = r.MolLogP.toFixed(2);
    }
    if(Number.isFinite(r.MolWt)) {
      ui.MolWt.value = r.MolWt;
      outputs.MolWt.textContent = r.MolWt.toFixed(1);
    }
    if(Number.isFinite(r.TPSA)) {
      ui.TPSA.value = r.TPSA;
      outputs.TPSA.textContent = r.TPSA.toFixed(1);
    }
    if(Number.isFinite(r.FC)) {
      ui.FC.value = r.FC;
      outputs.FC.textContent = r.FC.toString();
    }
    if(Number.isFinite(r.Aromatic)) {
      ui.Aromatic.value = r.Aromatic;
      outputs.Aromatic.textContent = r.Aromatic.toString();
    }
    if(Number.isFinite(r.Heavy)) {
      ui.Heavy.value = r.Heavy;
      outputs.Heavy.textContent = r.Heavy.toString();
    }
    
    // Log new values after update
    log('Slider values AFTER update:');
    log('  MolLogP:', oldValues.MolLogP, '->', ui.MolLogP.value);
    log('  MolWt:', oldValues.MolWt, '->', ui.MolWt.value);
    log('  TPSA:', oldValues.TPSA, '->', ui.TPSA.value);
    log('  FC:', oldValues.FC, '->', ui.FC.value);
    log('  Aromatic:', oldValues.Aromatic, '->', ui.Aromatic.value);
    log('  Heavy:', oldValues.Heavy, '->', ui.Heavy.value);
  }

  function updateLogBBDisplay(){
    const features = {
      MolLogP: +ui.MolLogP.value,
      MolWt:   +ui.MolWt.value,
      TPSA:    +ui.TPSA.value,
      FC:      +ui.FC.value,
      Aromatic:+ui.Aromatic.value,
      Heavy:   +ui.Heavy.value
    };
    const pred = predictLogBBFromFeatures(features);
    const selectedRow = findRowByName(drugSel.value);
    const exp = (selectedRow && Number.isFinite(selectedRow.logBB_exp)) ? selectedRow.logBB_exp : null;
    modelLogbbEl.textContent = Number.isFinite(pred) ? pred.toFixed(3) : '—';
    expLogbbEl.textContent = exp !== null ? exp.toFixed(3) : '—';
    return {pred, exp};
  }

  // plotting + run
  function runSimulation(){
    const {pred, exp} = updateLogBBDisplay();
    const P_eff = toNum(ui.P_eff.value) || 0.15;
    const Tmax  = Math.max(1, toNum(ui.Tmax.value) || 24);
    const dt = 0.05;

    const traces = [];

    if(Number.isFinite(pred)){
      const sim = simulateTwoComp(pred, P_eff, Tmax, dt);
      traces.push({ x: sim.t, y: sim.Cb, name: 'Blood (Model)', mode:'lines', line:{width:2, color:'#1f77b4'} });
      traces.push({ x: sim.t, y: sim.Cbr, name: 'Brain (Model)', mode:'lines', line:{width:2, color:'#ff7f0e'} });
    }

    if(Number.isFinite(exp)){
      const sim2 = simulateTwoComp(exp, P_eff, Tmax, dt);
      traces.push({ x: sim2.t, y: sim2.Cb, name: 'Blood (Experimental)', mode:'lines', line:{dash:'dot', width:2, color:'#1f77b4'} });
      traces.push({ x: sim2.t, y: sim2.Cbr, name: 'Brain (Experimental)', mode:'lines', line:{dash:'dot', width:2, color:'#ff7f0e'} });
    }

    const layout = {
      title: { text: `Fick's Law Simulation — Model LogBB=${Number.isFinite(pred)?pred.toFixed(3):'—'}  Exp LogBB=${(exp!==null)?exp.toFixed(3):'—'}`, font:{size:15} },
      xaxis: { title: 'Time (h)' },
      yaxis: { title: 'Concentration (normalized)', range:[0, 1.05] },
      legend: { orientation:'h', y:1.08 }
    };

    Plotly.react('plot', traces, layout, {responsive:true});
  }

  // events: dropdown -> update sliders & run
  drugSel.addEventListener('change', () => {
    log('=== DROPDOWN CHANGED to:', drugSel.value, '===');
    const r = findRowByName(drugSel.value);
    log('Found row:', r);
    if(r) applyRowToSliders(r);
    updateLogBBDisplay();
    runSimulation();
  });

  runBtn.addEventListener('click', () => {
    log('Run button clicked');
    runSimulation();
  });
  
  resetBtn.addEventListener('click', () => {
    log('Reset button clicked');
    const r = findRowByName(drugSel.value);
    if(r) applyRowToSliders(r);
    updateLogBBDisplay();
  });

  // initialize with first drug if present
  if(names.length > 0){
    log('Initializing with first drug:', names[0]);
    drugSel.value = names[0];
    const r = findRowByName(names[0]);
    if(r) applyRowToSliders(r);
  }

  // initial plot
  runSimulation();

})();
