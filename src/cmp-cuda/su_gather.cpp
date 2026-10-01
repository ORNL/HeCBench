////////////////////////////////////////////////////////////////////////////////
/**
 * @file su_gather.cpp
 * @date 2017-03-06
 * @author Tiago Lobato Gimenes    (tlgimenes@gmail.com)
 *
 * @copyright
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
////////////////////////////////////////////////////////////////////////////////

#include "su_gather.hpp"

#include "log.hpp"
#include "su_trace.hpp"

#include <algorithm>
#include <cassert>

////////////////////////////////////////////////////////////////////////////////

su_gather::su_gather(std::string& file_path, int aph, int nc) :
  _ttraces(0), _nos(0), _ns(-1), _ntrs(0)
{
  std::ifstream file(file_path, std::ios::binary);
  int min_ns = -1, max_ns = -1, max_ntrs = 0;
  std::map<int, su_cdp> traces_by_cdp;

//  su_trace tr;
//  for(int i=0; i < 50; i++) {
//    tr.fgettr(file);
  for(su_trace tr; tr.fgettr(file);) {
    real hx = tr.halfoffset_x();
    real hy = tr.halfoffset_y();

    if (hx*hx + hy*hy > aph*aph) continue;
    if (min_ns < 1) min_ns = tr.ns(); // First trace.
    if (max_ns < 1) max_ns = tr.ns(); // First trace.
    if (min_ns > tr.ns()) min_ns = tr.ns();
    if (max_ns < tr.ns()) max_ns = tr.ns();

    traces_by_cdp[tr.cdp()].push_back(tr);
    max_ntrs = max_ntrs < traces_by_cdp[tr.cdp()].size() ? traces_by_cdp[tr.cdp()].size() : max_ntrs;
  }
  assert(min_ns == max_ns);

  this->_ns = max_ns;
  this->_ntrs = max_ntrs & 0x1 ? max_ntrs + 1 : max_ntrs;

  _cdps.reserve(traces_by_cdp.size());
  for(auto& cdp: traces_by_cdp) {
    _cdps.push_back(cdp.second);
    _ttraces += cdp.second.traces().size();
  }
  this->_nos = _ttraces * _ns * nc;

  std::sort(_cdps.begin(), _cdps.end());
}

////////////////////////////////////////////////////////////////////////////////

void su_gather::linearize(int*& ntraces_by_cdp_id, real *&samples, real &dt, real* &gx, real* &gy, real* &sx, real* &sy, real* &scalco, int nc) {
  gx = new real[_ttraces];
  gy = new real[_ttraces];
  sx = new real[_ttraces];
  sy = new real[_ttraces];
  scalco = new real[_ttraces];
  samples = new real[_ttraces * _ns];
  ntraces_by_cdp_id = new int[_cdps.size()];

  dt = _cdps[0].traces()[0].dt();
  assert(dt > 0);

  int n_traces = 0;
  for(int i=0; i < _cdps.size(); i++) {
    ntraces_by_cdp_id[i] = _cdps[i].traces().size();
    ntraces_by_cdp_id[i] += (i > 0) ? ntraces_by_cdp_id[i-1] : 0;

    for(int j=0; j < _cdps[i].traces().size(); j++, n_traces++) {
      assert(n_traces < _ttraces);

      const su_trace& tr = _cdps[i].traces()[j];

      gx[n_traces] = tr.gx();
      gy[n_traces] = tr.gy();
      sx[n_traces] = tr.sx();
      sy[n_traces] = tr.sy();
      scalco[n_traces] = tr.scalco();

      for(int k=0; k < _ns; k++) {
        assert(tr.data().size() == _ns);
        samples[n_traces*_ns + k] = tr.data()[k];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
