import { Route, Routes } from "react-router-dom";
//import { MainNavigation } from "./components/layout/MainNavigation";
import { Prediction } from "./pages/Prediction";
import { Spectogram } from "./pages/Spectogram";
import { Home } from "./pages/Home";
import { Aboutus } from "./pages/Aboutus";

import { Layout } from "./components/layout/Layout";
function App() {
  return (
    <Layout className="">
    {/* // <MainNavigation> */}
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/prediction" element={<Prediction />} />
      <Route path="/visual" element={<Spectogram />} />
      <Route path="/about-us" element={<Aboutus />} />
    </Routes>
    {/* </MainNavigation> */}
     </Layout>
  );
}

export default App;


