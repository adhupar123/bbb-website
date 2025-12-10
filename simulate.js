/* simulate.js
 - Loads B3DB_regression.tsv
 - Computes OLS regression for LogBB
 - Syncs sliders & dropdown
 - Runs a two-compartment Fick's Law simulation
 - Produces Plotly plot of blood vs. brain concentration
*/

(async function(){
  /***************************************
   *             DATA LOADING
   ***************************************/
  async function loadTSV(path){
    const res = await fetch(path);
    if(!res.ok) throw new Error("Could not fetch TSV at " + path);
    const txt = await res.text();

    const lines = txt.trim().split(/\r?\n/);
    const headers = lines[0].split("\t");

    const rows = lines.slice(1).map(line => {
      const cols = line.split("\t");
      const obj = {};
      headers.forEach((h,i) => obj[h] = cols[i]);
      return obj;
    });

    return { headers, rows };
  }

  function toNum(v){
    return (v === "" || v === undefined) ? NaN : +v;
  }

  /***************************************
   *         MATRIX OPERATIONS (OLS)
   ***************************************/
  function transpose(A){
    return A[0].map((_,i)=> A.map(row => row[i]));
  }

  function matMul(A,B){
    const m=A.length, n=A[0].length, p=B[0].length;
    const result = Array.from({length:m}, () => Array(p).fill(0));

    for(let i=0;i<m;i++){
      for(let k=0;k<n;k++){
        const aik = A[i][k];
        for(let j=0;j<p;j++){
          result[i][j] += aik * B[k][j];
        }
      }
    }
    return result;
  }

  function invertMatrix(M){
    const n = M.length;
    const A = M.map(r=> r.slice());
    const I = Array.from({length:n}, (_,i)=>
      Array.from({length:n}, (__,j)=> (i===j ? 1 : 0))
    );

    for(let i=0;i<n;i++){
      let pivot = A[i][i];
      if(Math.abs(pivot) < 1e-12){
        let swap = i+1;
        while(swap<n && Math.abs(A[swap][i]) < 1e-12) swap++;
        if(swap === n) throw new Error("Singular matrix");
        [A[i], A[swap]] = [A[swap], A[i]];
        [I[i], I[swap]] = [I[swap], I[i]];
        pivot = A[i][i];
      }

      for(let j=0;j<n;j++){
        A[i][j] /= pivot;
        I[i][j] /= pivot;
      }

      for(let r=0;r<n;r++){
        if(r === i) continue;
        const f = A[r][i];
        for(let c=0;c<n;c++){
          A[r][c] -= f * A[i][c];
          I[r][c] -= f * I[i][c];
        }
      }
    }

    return I;
  }

  function ols(X, y){
    const Xt = transpose(X);
    const XtX = matMul(Xt, X);
    const XtXinv = invertMatrix(XtX);
    const Xty = matMul(Xt, y.map(v => [v]));
    const betaMatrix = matMul(XtXinv, Xty);
    return betaMatrix.map(row => row[0]);
  }

  /***************************************
   *       SIMULATION (Fick's Law)
   ***************************************/
  function simulateTwoCompartment(logBB, P_eff, Tmax, dt){
    const K = Math.pow(10, logBB);

    const steps = Math.max(2, Math.floor(Tmax/dt)+1);
    const t = new Array(steps);
    const Cb = new Array(steps);
    const Cbr = new Array(steps);

    let cb = 1.0;
    let cbr = 0.0;

    t[0] = 0;
    Cb[0] = cb;
    Cbr[0] = cbr;

    function rhs([cB, cBr]){
      const dcb  = -P_eff * (cB - (cBr / K));
      const dcbr =  P_eff * (cB - (cBr / K));
      return [dcb, dcbr];
    }

    for(let i=1;i<steps;i++){
      const time = i * dt;

      // RK4
      const s0 = [cb, cbr];
      const k1 = rhs(s0);

      const s1 = [ cb + 0.5*dt*k1[0], cbr + 0.5*dt*k1[1] ];
      const k2 = rhs(s1);

      const s2 = [ cb + 0.5*dt*k2[0], cbr + 0.5*dt*k2[1] ];
      const k3 = rhs(s2);

      const s3 = [ cb + dt*k3[0], cbr + dt*k3[1] ];
      const k4 = rhs(s3);

      cb  = cb  + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
      cbr = cbr + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);

      t[i] = time;
      Cb[i] = cb;
      Cbr[i] = cbr;
    }

    return { t, Cb, Cbr };
  }

  /***************************************
   *           UI ELEMENTS
   ***************************************/
  const controlIds = ['MolLogP','MolWt','TPSA','FC','Aromatic','Heavy'];
  const ui = {};

  controlIds.forEach(id => ui[id] = document.getElementById(id));
  ui["P_eff"] = document.getElementById("P_eff");
  ui["Tmax"] = document.getElementById("Tmax");

  const outputEls = {
    MolLogP: document.getElementById("MolLogP_val"),
    MolWt: document.getElementById("MolWt_val"),
    TPSA: document.getElementById("TPSA_val"),
    FC: document.getElementById("FC_val"),
    Aromatic: document.getElementById("Aromatic_val"),
    Heavy: document.getElementById("Heavy_val")
  };

  const drugSelect = document.getElementById("drug-select");
  const runBtn = document.getElementById("runBtn");
  const resetBtn = document.getElementById("resetBtn");

  const modelLogbbEl = document.getElementById("model_logbb");
  const expLogbbEl = document.getElementById("exp_logbb");

  // update slider outputs live
  controlIds.forEach(id => {
    ui[id].addEventListener("input", () => {
      outputEls[id].textContent = ui[id].value;
    });
    outputEls[id].textContent = ui[id].value; // initial display
  });

  /***************************************
   *               LOAD TSV
   ***************************************/
  let tsv;
  try{
    tsv = await loadTSV("./B3DB_regression.tsv");
  } catch(e){
    console.error(e);
    document.getElementById("plot").innerHTML =
      "<p style='color:#c00'>Error loading B3DB_regression.tsv</p>";
    return;
  }

  const { rows, headers } = tsv;

  // Attempt flexible column matching
  function findColumn(candidates){
    for(const c of candidates){
      if(headers.includes(c)) return c;
      const match = headers.find(h => h.toLowerCase().replace(/\s+/g,"") === c.toLowerCase().replace(/\s+/g,""));
      if(match) return match;
    }
    return null;
  }

  const colName      = findColumn(["compound_name","compound","name"]);
  const colMolLogP   = findColumn(["MolLogP","LogP"]);
  const colMolWt     = findColumn(["MolWt","MolecularWeight"]);
  const colTPSA      = findColumn(["TPSA"]);
  const colFC        = findColumn(["FC","FormalCharge"]);
  const colAromatic  = findColumn(["Aromatic Rings","Aromatic"]);
  const colHeavy     = findColumn(["Heavy Atoms","HeavyAtoms"]);
  const colLogBB     = findColumn(["logBB","logBB_exp","LogBB"]);

  const parsedRows = rows.map(r => ({
    name:     r[colName],
    MolLogP:  toNum(r[colMolLogP]),
    MolWt:    toNum(r[colMolWt]),
    TPSA:     toNum(r[colTPSA]),
    FC:       toNum(r[colFC]),
    Aromatic: toNum(r[colAromatic]),
    Heavy:    toNum(r[colHeavy]),
    logBB_exp: colLogBB ? toNum(r[colLogBB]) : NaN
  }));

  // Populate dropdown
  const uniqueNames = Array.from(new Set(parsedRows.map(r => r.name))).filter(Boolean).sort();
  uniqueNames.forEach(n => {
    const opt = document.createElement("option");
    opt.value = n;
    opt.textContent = n;
    drugSelect.appendChild(opt);
  });

  /***************************************
   *       PREPARE DATA FOR OLS
   ***************************************/
  const X = [];
  const y = [];

  parsedRows.forEach(r => {
    if([r.MolLogP, r.MolWt, r.TPSA, r.FC, r.Aromatic, r.Heavy].some(v => Number.isNaN(v))) return;
    X.push([1, r.MolLogP, r.MolWt, r.TPSA, r.FC, r.Aromatic, r.Heavy]);
    y.push(r.logBB_exp);
  });

  // keep only rows with known logBB for training
  const X_train = [], y_train = [];
  for(let i=0;i<X.length;i++){
    if(!Number.isNaN(y[i])){
      X_train.push(X[i]);
      y_train.push(y[i]);
    }
  }

  let beta;
  try{
    beta = ols(X_train, y_train);
  } catch(e){
    console.error("OLS failed:", e);
    beta = null;
  }

  function predictLogBB(features){
    if(!beta) return NaN;
    const vals = [1, features.MolLogP, features.MolWt, features.TPSA, features.FC, features.Aromatic, features.Heavy];
    return vals.reduce((sum, v, i) => sum + v*beta[i], 0);
  }

  function getDrugRow(name){
    return parsedRows.find(r => r.name === name);
  }

  function setSlidersFromRow(r){
    ui.MolLogP.value = r.MolLogP;
    ui.MolWt.value   = r.MolWt;
    ui.TPSA.value    = r.TPSA;
    ui.FC.value      = r.FC;
    ui.Aromatic.value = r.Aromatic;
    ui.Heavy.value   = r.Heavy;

    controlIds.forEach(id => outputEls[id].textContent = ui[id].value);
  }

  /***************************************
   *           UPDATE TEXT LABELS
   ***************************************/
  function updateModelDisplay(){
    const features = {
      MolLogP: +ui.MolLogP.value,
      MolWt:   +ui.MolWt.value,
      TPSA:    +ui.TPSA.value,
      FC:      +ui.FC.value,
      Aromatic:+ui.Aromatic.value,
      Heavy:   +ui.Heavy.value
    };

    const predicted = predictLogBB(features);

    const drug = getDrugRow(drugSelect.value);
    const exp  = (drug && Number.isFinite(drug.logBB_exp)) ? drug.logBB_exp : null;

    modelLogbbEl.textContent = Number.isFinite(predicted) ? predicted.toFixed(2) : "—";
    expLogbbEl.textContent   = exp !== null ? exp.toFixed(2) : "—";

    return { predicted, exp };
  }

  /***************************************
   *               PLOTTING
   ***************************************/
  function runSimulation(){
    const { predicted, exp } = updateModelDisplay();
    const P_eff = +ui.P_eff.value || 0.15;
    const Tmax  = +ui.Tmax.value || 24;
    const dt = 0.05;

    const traces = [];

    if(Number.isFinite(predicted)){
      const out = simulateTwoCompartment(predicted, P_eff, Tmax, dt);
      traces.push({
        x: out.t, y: out.Cb, mode:"lines",
        name:"Blood (Model)", line:{width:2}
      });
      traces.push({
        x: out.t, y: out.Cbr, mode:"lines",
        name:"Brain (Model)", line:{width:2}
      });
    }

    if(Number.isFinite(exp)){
      const outExp = simulateTwoCompartment(exp, P_eff, Tmax, dt);
      traces.push({
        x: outExp.t, y: outExp.Cb, mode:"lines",
        name:"Blood (Experimental)", line:{dash:"dash", width:2}
      });
      traces.push({
        x: outExp.t, y: outExp.Cbr, mode:"lines",
        name:"Brain (Experimental)", line:{dash:"dash", width:2}
      });
    }

    const layout = {
      title: "Fick's Law Simulation",
      xaxis: { title: "Time (h)" },
      yaxis: { title: "Concentration (normalized)", range: [0,1.05] },
      legend: { orientation:"h" },
      margin: { t:50 }
    };

    Plotly.newPlot("plot", traces, layout, {responsive:true});
  }

  /***************************************
   *       INITIALIZATION & EVENTS
   ***************************************/
  drugSelect.addEventListener("change", () => {
    const row = getDrugRow(drugSelect.value);
    if(row) setSlidersFromRow(row);
    updateModelDisplay();
  });

  runBtn.addEventListener("click", runSimulation);

  resetBtn.addEventListener("click", () => {
    const row = getDrugRow(drugSelect.value);
    if(row) setSlidersFromRow(row);
    updateModelDisplay();
  });

  // Set default drug
  if(uniqueNames.length > 0){
    drugSelect.value = uniqueNames[0];
    const row = getDrugRow(uniqueNames[0]);
    setSlidersFromRow(row);
    updateModelDisplay();
  }

  runSimulation();

})();
