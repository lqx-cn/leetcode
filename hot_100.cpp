#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <limits.h>
#include <algorithm>
using namespace std;
/**
 * Definition for a binary tree node.
 */
struct TreeNode
{
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution
{
public:
  /*
  189. 轮转数组
  input: [1, 2, 3, 4, 5, 6] k = 2
  output: [5, 6, 1, 2, 3, 4]
  */
  void rotate(vector<int> &nums, int k)
  {
    int n = nums.size();
    int real_k = k % n;
    vector<int> tmp(real_k);
    for (int i = 0; i < real_k; i++)
    {
      tmp[real_k - 1 - i] = nums[n - 1 - i];
    }
    for (int i = 0; i < n; i++)
    {
      if (n - i - 1 - real_k >= 0)
      {
        nums[n - i - 1] = nums[n - i - 1 - real_k];
      }
      else
      {
        nums[n - i - 1] = tmp[n - i - 1];
      }
    }
  }
  /*
  238. 除自身以外数组的乘积
  */
  vector<int> productExceptSelf(vector<int> &nums)
  {
    vector<int> res(nums.size());
    int multiple_res = 1;
    int zero_cnt = 0;
    for (int i = 0; i < nums.size(); i++)
    {
      if (nums[i] != 0)
      {
        multiple_res *= nums[i];
      }
      else
      {
        zero_cnt++;
      }
    }
    for (int i = 0; i < nums.size(); i++)
    {
      if (nums[i] == 0)
      {
        if (zero_cnt == 1)
        {
          res[i] = multiple_res;
        }
        else
        {
          res[i] = 0;
        }
      }
      else
      {
        if (zero_cnt == 0)
        {
          res[i] = multiple_res / nums[i];
        }
        else
        {
          res[i] = 0;
        }
      }
    }
    return res;
  }

  /*
    3. 无重复字符的最长子串
    输入：s = "abcabcbb"
    输出：3 因为 abc为最长
    思路：
      1. （滑动窗口的特点就是右扩左缩），设置双指针表示滑动窗口的左右两侧
      2. 当s[r] 没有出现过时，长度++
      3. 当s[r] 出现过，那么得移动l，直至s[r]没有出现
  */
  int lengthOfLongestSubstring(string s)
  {
    if (s.size() < 2)
    {
      return s.size();
    }

    int l = 0, r = 1;
    int max_length = 1;
    int length = 1;
    unordered_set<char> set;
    set.insert(s[0]);
    while (r < s.size())
    {
      if (set.count(s[r]) == 0)
      {
        length++;
        set.insert(s[r]);
        max_length = max(length, max_length);
        r++;
      }
      else
      {
        length--;
        set.erase(s[l]);
        l++;
      }
    }
    return max_length;
  }

  /*
  42. 接雨水
  */
  int trap(vector<int> &height)
  {
    int n = height.size();
    if (n < 3)
    {
      return 0;
    }
    vector<int> left_heigher_idx(n, -1);
    vector<int> right_heigher_idx(n, -1);
    deque<int> st;
    for (int i = 0; i < n; i++)
    {
      while (!st.empty() && height[i] >= height[st.back()])
      {
        st.pop_back();
      }

      if (!st.empty())
      {
        left_heigher_idx[i] = st.back();
      }
      else
      {
        st.push_back(i);
      }
    }
    st.clear();
    for (int i = n - 1; i > -1; i--)
    {
      while (!st.empty() && height[i] >= height[st.back()])
      {
        st.pop_back();
      }
      if (!st.empty())
      {
        right_heigher_idx[i] = st.back();
      }
      else
      {
        st.push_back(i);
      }
    }
    deque<int>(0).swap(st);
    int capacity = 0;
    for (int i = 0; i < n; i++)
    {
      if (left_heigher_idx[i] != -1 && right_heigher_idx[i] != -1)
      {
        capacity += (min(height[left_heigher_idx[i]], height[right_heigher_idx[i]]) - height[i]);
      }
    }
    return capacity;
  }

  /*
  15. 三数之和为0
  注意：本题最终要的是避免枚举重复元素
  思路：
    1. 异常输入返回（小于三个元素的输入直接返回）
    2. 对数组进行排序，然后开始枚举第一个元素，注意不能重复
    3. 使用双指针
  */
  vector<vector<int>> threeSum(vector<int> &nums)
  {
    vector<vector<int>> res;
    int n = nums.size();
    if (n < 3)
    {
      return res;
    }
    sort(nums.begin(), nums.end());
    for (int i = 0; i < n; i++)
    {
      if (nums[i] > 0)
      {
        break;
      }
      if (i > 0 && nums[i] == nums[i - 1])
      {
        continue;
      }
      int l = i + 1, r = n - 1;
      while (l < r)
      {
        if (l > i + 1 && nums[l] == nums[l - 1])
        {
          l++;
          continue;
        }
        if (r < n - 1 && nums[r] == nums[r + 1])
        {
          r--;
          continue;
        }
        if (nums[l] + nums[r] == -nums[i])
        {
          res.push_back({nums[i], nums[l], nums[r]});
          l++;
          r--;
        }
        else if (nums[l] + nums[r] < -nums[i])
        {
          l++;
        }
        else
        {
          r--;
        }
      }
    }
    return res;
  }

  /*
  46. 全排列
  输入：nums = [1,2,3]
  输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
  思路：回溯+递归
  method1: 在原来的数组 nums 上进行更改，用一个标志位first来表示当前需要改第几个元素，
           然后遍历first之后的数组，将第i个数和first进行交换
  method2: 每一次递归选一个元素，需要维护一个set来看元素有没有被访问过
  */
  void backTrack1(vector<int> &nums, int first, int len, vector<vector<int>> &res)
  {
    if (first == len)
    {
      res.push_back(nums);
    }
    for (int i = first; i < len; i++)
    {
      swap(nums[first], nums[i]);
      backTrack1(nums, first + 1, len, res);
      swap(nums[first], nums[i]);
    }
  }
  void backTrack2(const vector<int> &nums, vector<int> curr, vector<vector<int>> &res, unordered_set<int> &is_pass)
  {
    if (is_pass.size() == nums.size())
    {
      res.push_back(curr);
      return;
    }
    for (auto num : nums)
    {
      if (is_pass.count(num) == 1)
      {
        continue;
      }
      curr.push_back(num);
      is_pass.insert(num);
      backTrack2(nums, curr, res, is_pass);
      curr.pop_back();
      is_pass.erase(num);
    }
  }

  vector<vector<int>> permute(vector<int> &nums)
  {
    vector<vector<int>> res;
    // method 1
    backTrack1(nums, 0, nums.size(), res);

    // method 2
    // vector<int> curr;
    // unordered_set<int> is_pass;
    // backTrack2(nums, curr, res, is_pass);
    return res;
  }

  /*
  49. 字母异位词分组
  输入：['tea', 'tae', 'hello', 'lolhe']
  期望输出：[['tea', 'tae'], ['hello', 'lolhe']]
  思路：
    1. 将每个strs中的str进行排序，这样就可以将异位的字符串变为相等的
    2. 然后将这个相等的字符串作为keys值，就可以找到相应的分组了
  */
  vector<vector<string>> groupAnagrams(vector<string> &strs)
  {
    vector<vector<string>> res;
    if (strs.size() == 0)
    {
      return res;
    }
    vector<unordered_map<char, int>> counts(strs.size());
    for (int i = 0; i < strs.size(); i++)
    {
      auto &str = strs[i];
      auto &count = counts[i];
      for (auto c : str)
      {
        auto iter = count.find(c);
        if (iter != count.end())
        {
          iter->second++;
        }
        else
        {
          count.insert(pair<char, int>(c, 1));
        }
      }
    }
    vector<bool> is_traversed(strs.size(), false);

    for (int i = 0; i < strs.size(); i++)
    {
      if (is_traversed[i])
      {
        continue;
      }
      vector<string> tmp;
      tmp.push_back(strs[i]);
      is_traversed[i] = true;
      auto &count_i = counts[i];
      for (int j = 0; j < strs.size(); j++)
      {
        if (is_traversed[j])
        {
          continue;
        }
        auto &count_j = counts[j];
        bool is_same = true;
        if (count_i.size() != count_j.size())
        {
          continue;
        }
        for (auto iter = count_i.begin(); iter != count_i.end(); iter++)
        {
          if (count_j.find(iter->first) == count_j.end())
          {
            is_same = false;
            break;
          }
          else if (count_j.find(iter->first)->second != iter->second)
          {
            is_same = false;
            break;
          }
        }
        if (is_same)
        {
          tmp.push_back(strs[j]);
          is_traversed[j] = true;
        }
      }
      res.push_back(tmp);
    }
    return res;
  }

  /*
  53. 最大子数组和
  输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
  输出：6
  解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
  */
  int maxSubArray(vector<int> &nums)
  {
    int n = nums.size();
    if (n == 0)
    {
      return 0;
    }
    int max_sum = INT_MIN;
    vector<int> dp(n + 1);
    dp[0] = INT_MIN;
    dp[1] = nums[0];
    for (int i = 2; i <= n; i++)
    {
      if (dp[i - 1] < 0)
      {
        dp[i] = nums[i - 1];
      }
      else
      {
        dp[i] = dp[i - 1] + nums[i - 1];
      }
    }
    for (auto e : dp)
    {
      max_sum = max(e, max_sum);
    }

    return max_sum;
  }
  /*
  78. 子集
  输入：nums = [1,2,3]
  输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
  */
  void subset_trackback(vector<int> &nums, int len, int start, int first, vector<vector<int>> &res)
  {
    if (first == len)
    {
      vector<int> tmp(nums.begin(), nums.begin() + len);
      res.push_back(tmp);
    }

    for (int i = start; i < nums.size(); i++)
    {
      swap(nums[first], nums[i]);
      subset_trackback(nums, len, i + 1, first + 1, res);
      swap(nums[first], nums[i]);
    }
  }
  vector<vector<int>> subsets(vector<int> &nums)
  {
    vector<vector<int>> res;
    int n = nums.size();
    for (int l = 0; l <= n; l++)
    {
      subset_trackback(nums, l, 0, 0, res);
    }
    return res;
  }
  /*
  101. 对称二叉树
  */
  bool isSame(TreeNode *left, TreeNode *right)
  {
    if (left == nullptr && right == nullptr)
    {
      return true;
    }
    else if (left == nullptr || right == nullptr)
    {
      return false;
    }
    if (left->val == right->val)
    {
      return isSame(left->right, right->left) && isSame(left->left, right->right);
    }
    else
    {
      return false;
    }
  }
  bool isSymmetric(TreeNode *root)
  {
    return isSame(root->left, root->right);
  }
  /*
  102. 二叉树的层序遍历
  输入：root = [3,9,20,null,null,15,7]
  输出：[[3],[9,20],[15,7]]
  思路：用两个while循环和两个队列，装不同层的节点
  */
  vector<vector<int>> levelOrder(TreeNode *root)
  {
    if (root == nullptr)
    {
      return vector<vector<int>>();
    }
    queue<TreeNode *> que1;
    queue<TreeNode *> que2;
    que1.push(root);
    vector<vector<int>> res;
    vector<int> res_tmp;

    while (!que1.empty())
    {
      TreeNode *curr = que1.front();

      if (curr->left != nullptr)
      {
        que2.push(curr->left);
      }

      if (curr->right != nullptr)
      {
        que2.push(curr->right);
      }
      res_tmp.push_back(curr->val);
      que1.pop();
      if (que1.empty() && !que2.empty())
      {
        res.push_back(res_tmp);
        res_tmp.clear();
        while (!que2.empty())
        {
          TreeNode *next_curr = que2.front();
          if (next_curr->left != nullptr)
          {
            que1.push(next_curr->left);
          }

          if (next_curr->right != nullptr)
          {
            que1.push(next_curr->right);
          }
          que2.pop();
          res_tmp.push_back(next_curr->val);
        }
        res.push_back(res_tmp);
        res_tmp.clear();
      }
    }
    return res;
  }

  /*
  128. 最长连续序列
  输入：[3, 2, 1, -1, 100]
  输出：3 即 (1, 2, 3) 为最长连续序列
  思路：
    1. 创建一个 unorder_set，将数组元素放进里面方便查找
    2. 开始遍历 unorder_set，如果 num - 1 存在，就说明这不是开头的数，直接跳过
    3. 不断循环 num + 1 ，当 num + 1 一直存在的时候，长度++，表示连续序列存在
  */
  int longestConsecutive(vector<int> &nums)
  {
    if (nums.size() <= 1)
    {
      return nums.size();
    }
    unordered_set<int> ms;
    for (int i = 0; i < nums.size(); i++)
    {
      ms.insert(nums[i]);
    }

    int max_length = -1;

    for (int num : ms)
    {
      if (ms.count(num - 1) == 0)
      {
        int curr_num = num;
        int length = 0;
        while (ms.count(curr_num) > 0)
        {
          curr_num++;
          length++;
        }
        max_length = max(length, max_length);
      }
    }
    return max_length;
  }

  /*
  139. 单词拆分
  输入: s = "leetcode", wordDict = ["leet", "code"]
  输出: true
  解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
  思路：dp[i] = dp[i-j] && s.substr(i-j, j) for j
  */
  bool wordBreak(string s, vector<string> &wordDict)
  {
    unordered_set<string> word_set;
    for (const auto &s : wordDict)
    {
      word_set.insert(s);
    }
    vector<bool> dp(s.size() + 1);
    dp[0] = true;
    for (int i = 1; i <= s.size(); i++)
    {
      dp[i] = false;
      for (int j = 0; j < i; j++)
      {
        bool flag = word_set.count(s.substr(j, i - j)) == 1;
        dp[i] = dp[i] || (dp[j] && flag);
      }
    }
    return dp[s.size()];
  }

  /*
  152. 乘积最大子数组
  输入: nums = [2,3,-2,4]
  输出: 6
  解释: 子数组 [2,3] 有最大乘积 6。
  思路：弄两个dp，一个维护乘积最大，一个维护乘积最小，要考虑0的影响
  */
  int maxProduct(vector<int> &nums)
  {
    int n = nums.size();
    vector<int> max_dp(n), min_dp(n);
    max_dp[0] = min_dp[0] = nums[0];
    for (int i = 1; i < n; i++)
    {
      max_dp[i] = max(max(max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i]), nums[i]);
      min_dp[i] = min(min(max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i]), nums[i]);
    }
    return *max_element(max_dp.begin(), max_dp.end());
  }

  /*
  198. 打家劫舍
  输入：[1,2,3,1]
  输出：4
  解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。偷窃到的最高金额 = 1 + 3 = 4 。
  思路：状态转移方程： dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
  */
  int rob(vector<int> &nums)
  {
    int n = nums.size();
    vector<int> dp(n);
    if (n == 1)
    {
      return nums[0];
    }
    dp[0] = nums[0];
    dp[1] = max(nums[0], nums[1]);
    for (int i = 2; i < n; i++)
    {
      dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
    }
    return dp[n - 1];
  }

  /*
  226. 翻转二叉树
  */
  TreeNode *invertTree(TreeNode *root)
  {
    if (root == nullptr)
    {
      return nullptr;
    }
    TreeNode *left = invertTree(root->left);
    TreeNode *right = invertTree(root->right);
    root->left = right;
    root->right = left;
    return root;
  }
  /*
  279. 完全平方数
  给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
  思路：dp[i] = min(dp[i-n^2] + 1)
  */
  int numSquares(int n)
  {
    vector<int> dp(n + 1);
    dp[0] = 0;
    dp[1] = 1;
    for (int i = 2; i <= n; i++)
    {
      int tmp = INT_MAX;
      for (int j = 1; j * j <= i; j++)
      {
        tmp = min(dp[i - j * j] + 1, tmp);
      }
      dp[i] = tmp;
    }
    return dp[n];
  }

  /*
  283. 将所有的0移动到数组末尾
  */
  void moveZeroes(vector<int> &nums)
  {
    if (nums.size() < 2)
    {
      return;
    }
    int idx = 0;
    int stp = 0;
    while (idx < nums.size())
    {
      if (nums[idx] == 0)
      {
        stp++;
        idx++;
        continue;
      }
      nums[idx - stp] = nums[idx];
      idx++;
    }
    for (int i = 0; i < stp; i++)
    {
      nums[nums.size() - 1 - i] = 0;
    }
  }

  /*
  300. 最长递增子序列
  输入：nums = [10,9,2,5,3,7,101,18]
  输出：4
  解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
  思路：dp[i] 表示 以前i个元素结尾的最长递增子序列长度
  */
  int lengthOfLIS(vector<int> &nums)
  {
    int n = nums.size();
    vector<int> dp(n + 1);
    dp[0] = 0;
    dp[1] = 1;
    for (int i = 2; i <= n; i++)
    {
      dp[i] = INT_MIN;
      for (int j = 1; j < i; j++)
      {
        dp[i] = max(dp[i], nums[j - 1] < nums[i - 1] ? dp[j] + 1 : 1);
      }
    }
    return *max_element(dp.begin(), dp.end());
  }

  /*
  322. 零钱兑换
  输入：coins = [1, 2, 5], amount = 11
  输出：3
  解释：11 = 5 + 5 + 1
  思路：dp[i] = min(dp[i-coins[j]] + 1) 遍历所有的 coins
  */
  int coinChange(vector<int> &coins, int amount)
  {
    vector<int> dp(amount + 1);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++)
    {
      dp[i] = -1;
      int tmp = INT_MAX;
      for (int j = 0; j < coins.size(); j++)
      {
        if (i - coins[j] < 0)
        {
          continue;
        }
        if (dp[i - coins[j]] == -1)
        {
          continue;
        }
        tmp = min(dp[i - coins[j]] + 1, tmp);
      }
      if (tmp < INT_MAX)
      {
        dp[i] = tmp;
      }
    }
    return dp[amount];
  }

  /*
   438. 找到字符串中所有字母异位词
   输入: s = "cbaebabacd", p = "abc"
   输出: [0,6]
   解释:
   起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
   起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
   */
  vector<int> findAnagrams(string s, string p)
  {
    vector<int> res;
    if (s.size() < p.size())
    {
      return res;
    }
    vector<int> count_s(26, 0);
    vector<int> count_p(26, 0);
    for (int i = 0; i < p.size(); i++)
    {
      count_p[p[i] - 'a']++;
    }

    for (int i = 0; i < s.size(); i++)
    {
      count_s[s[i] - 'a']++;
      int l = i - p.size() + 1;
      if (l < 0)
      {
        continue;
      }
      else if (count_s == count_p)
      {
        res.push_back(l);
      }
      count_s[s[l] - 'a']--;
    }
    return res;
  }

  /*
  560. 和为 K 的子数组
  给你一个整数数组 nums 和一个整数 k ，请你统计并返回该数组中和为 k 的子数组的个数 。
  子数组是数组中元素的!连续!非空序列。
  input: [1, 2, 3] , k = 3
  output: 2 [[1, 2], [3]]
  */
  // method 2: front nums sum
  int subarraySum(vector<int> &nums, int k)
  {
    int cnt = 0;
    int sum = 0;
    unordered_map<int, int> front_sums;
    for (int i = 0; i < nums.size(); i++)
    {
      sum += nums[i];
      if (sum == k)
      {
        cnt++;
      }
      auto iter = front_sums.find(sum - k);
      if (iter != front_sums.end())
      {
        cnt += iter->second;
      }

      iter = front_sums.find(sum);
      if (iter != front_sums.end())
      {
        iter->second++;
      }
      else
      {
        front_sums[sum] = 1;
      }
    }
    return cnt;
  }
  // method 1: track back; overtime
  //  void getSumTraceBack(vector<int> &nums, int &curr_sum, int first, int k, int &cnt)
  //  {
  //    if (curr_sum == k)
  //    {
  //      cnt++;
  //    }

  //   if (first > nums.size() - 1)
  //   {
  //     return;
  //   }
  //   curr_sum += nums[first];
  //   getSumTraceBack(nums, curr_sum, first + 1, k, cnt);
  //   curr_sum -= nums[first];
  // }
  // int subarraySum(vector<int> &nums, int k)
  // {
  //   int curr_sum = 0;
  //   int cnt = 0;
  //   sort(nums.begin(), nums.end());
  //   for (int s = 0; s < nums.size(); s++)
  //   {
  //     curr_sum += nums[s];
  //     getSumTraceBack(nums, curr_sum, s + 1, k, cnt);
  //     curr_sum -= nums[s];
  //   }
  //   return cnt;
  // }

  void insert(TreeNode *root, int num)
  {
    if (root == nullptr)
    {
      root = new TreeNode(num);
      return;
    }
    TreeNode *curr = root, *pre = nullptr;
    while (curr)
    {
      if (curr->val == num)
      {
        return;
      }
      pre = curr;
      if (curr->val < num)
      {
        curr = curr->right;
      }
      else
      {
        curr = curr->left;
      }
    }
    if (pre->val < num)
    {
      pre->right = new TreeNode(num);
    }
    else
    {
      pre->left = new TreeNode(num);
    }
  }
  TreeNode *sortedArrayToBST(vector<int> &nums)
  {
    int root_idx = nums.size() / 2;
    auto iter = nums.begin() + root_idx - 1;
    TreeNode *root = new TreeNode(nums[root_idx]);
    nums.erase(iter);
    for (auto n : nums)
    {
      insert(root, n);
    }
    return root;
  }
};

int main(void)
{
  Solution s;
  // vector<int> nums({-1, 0, 1, 2, -1, -4});
  // s.threeSum(nums);
  vector<string> wordDict = vector<string>({"leet", "code"});
  string str = "leetcode";
  s.wordBreak(str, wordDict);
}