#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <limits.h>
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
    while (l < r && r < s.size())
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
  15. 三数之和为0
  思路：
    1. 考虑到数组中有3个以上的0时，{0, 0, 0} 也是一个解，先装进res中
    2. 先从小到大排序，并找到第一个大于0的数，如果没有找到，直接返回res
    3. 设置左右指针l, r； l从左到mid，r从右到mid，进行遍历
    4. 根据 nums[l] 和 nums[r] 的绝对值大小，选择找第三个数，范围在 l+1 到 mid 或 mid 到 r-1 中
  */
  vector<vector<int>> threeSum(vector<int> &nums)
  {
    vector<vector<int>> res;
    sort(nums.begin(), nums.end());
    auto iter = find_if(nums.begin(), nums.end(), [](int num)
                        { return num > 0; });
    int count_0 = 0;
    for (auto n : nums)
    {
      if (n == 0)
      {
        count_0++;
      }
    }
    if (count_0 > 2)
    {
      res.push_back(vector<int>({0, 0, 0}));
    }

    int mid = distance(nums.begin(), iter);
    if (mid == 0 || mid == nums.size())
    {
      return res;
    }

    for (int l = 0; l < mid; l++)
    {
      if (l > 0 && nums[l] == nums[l - 1])
      {
        continue;
      }
      for (int r = nums.size() - 1; r >= mid; r--)
      {
        if (r < nums.size() - 1 && nums[r] == nums[r + 1])
        {
          continue;
        }
        if (abs(nums[l]) > abs(nums[r]))
        {
          for (int i = mid; i < r; i++)
          {
            if (nums[i] == -(nums[r] + nums[l]))
            {
              res.push_back(vector<int>({nums[l], nums[r], nums[i]}));
              break;
            }
            else if (nums[i] > -(nums[r] + nums[l]))
            {
              break;
            }
          }
        }
        else
        {
          for (int i = l + 1; i <= mid; i++)
          {
            if (nums[i] == -(nums[r] + nums[l]))
            {
              res.push_back(vector<int>({nums[l], nums[r], nums[i]}));
              break;
            }
            else if (nums[i] > -(nums[r] + nums[l]))
            {
              break;
            }
          }
        }
      }
    }
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
    int n = p.size();
    vector<int> res;
    if (s.size() < n)
    {
      return res;
    }
    sort(p.begin(), p.end());
    int idx = 0;
    int l = 0;
    while (l <= s.size() - n)
    {
      string tmp = s.substr(l, n);
      sort(tmp.begin(), tmp.end());
      if (tmp == p)
      {
        res.push_back(l);
      }
      l++;
    }
    return res;
  }

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