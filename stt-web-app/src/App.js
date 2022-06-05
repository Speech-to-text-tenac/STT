import { Route, Routes } from "react-router-dom";
//import { MainNavigation } from "./components/layout/MainNavigation";
import { Prediction } from "./pages/Prediction";
import { Spectogram } from "./pages/Spectogram";

import { Layout } from "./components/layout/Layout";
function App() {
  return (
    <Layout className="">
    {/* // <MainNavigation> */}
    <Routes>
      <Route path="/" element={<Prediction />} />
      <Route path="/spectogram" element={<Spectogram />} />
    </Routes>
    {/* </MainNavigation> */}
     </Layout>
  );
}

export default App;


