import React from 'react';
import {  Link } from "react-router-dom";
const navbar= () =>{
  return (
  <div>
    <li>
      <Link to="/">Record</Link>
    </li>
    <li>
      <Link to="/Amharic">Amharic</Link>
    </li>
    <li>
      <Link to="/Home">Home</Link>
    </li>
  </div>
  );
}
export default navbar;