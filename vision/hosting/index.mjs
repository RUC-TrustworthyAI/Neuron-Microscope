import {mediaRoute} from './media.mjs';

export default {
  async fetch(request, env) {
    return await mediaRoute(request, env) ?? env.ASSETS.fetch(request);
  }
};
